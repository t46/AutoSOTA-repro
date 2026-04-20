"""AgentFix -- structured failure repair.

Analyses tracebacks, looks up known error patterns, queries failure
memory, and generates repairs via LLM when no match is found.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from autosota.agents.base import AgentResult, BaseAgent


# ------------------------------------------------------------------
# Structured models
# ------------------------------------------------------------------


class FailureSignature(BaseModel):
    """Normalised failure signature for deduplication and lookup."""

    error_type: str = ""
    error_message: str = ""
    file_path: str = ""
    line_number: int | None = None
    hash: str = ""


class RepairAction(BaseModel):
    """A concrete repair to apply."""

    description: str
    commands: list[str] = Field(default_factory=list)
    code_changes: list[dict[str, str]] = Field(default_factory=list)
    confidence: float = 0.5  # 0-1


class KnownPattern(BaseModel):
    """A known error pattern with a standard fix."""

    pattern: str
    category: str  # "pip", "cuda", "path", "import", "permission", "oom"
    fix_template: str


# ------------------------------------------------------------------
# Built-in known patterns
# ------------------------------------------------------------------

KNOWN_PATTERNS: list[KnownPattern] = [
    KnownPattern(
        pattern=r"ModuleNotFoundError: No module named '(\w+)'",
        category="import",
        fix_template="pip install {match_1}",
    ),
    KnownPattern(
        pattern=r"FileNotFoundError:.*No such file or directory: '([^']+)'",
        category="path",
        fix_template="mkdir -p $(dirname {match_1})",
    ),
    KnownPattern(
        pattern=r"CUDA out of memory",
        category="oom",
        fix_template="Reduce batch size or enable gradient checkpointing",
    ),
    KnownPattern(
        pattern=r"CUDA error: device-side assert triggered",
        category="cuda",
        fix_template="Check tensor shapes and label indices",
    ),
    KnownPattern(
        pattern=r"RuntimeError: Expected .* but got .*",
        category="cuda",
        fix_template="Check device placement; ensure all tensors on same device",
    ),
    KnownPattern(
        pattern=r"PermissionError",
        category="permission",
        fix_template="chmod +x {file} or check file permissions",
    ),
    KnownPattern(
        pattern=r"pip.*Could not find a version that satisfies",
        category="pip",
        fix_template="Check package name and version constraints; try pip install --upgrade pip",
    ),
    KnownPattern(
        pattern=r"OSError: \[Errno 28\] No space left on device",
        category="disk",
        fix_template="Clean up temporary files or increase disk allocation",
    ),
]


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------


class AgentFix(BaseAgent):
    """Structured repair agent for pipeline failures.

    Workflow:
    1. Normalise the error into a reusable signature.
    2. Match against known patterns (fast, no LLM).
    3. Check failure memory for previous encounters.
    4. If no match, generate a repair via LLM.
    5. Record the outcome for future reference.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._memory_path = self.workspace_dir / "memory" / "failure_memory.json"

    async def run(
        self,
        *,
        failure: dict[str, Any],
        env: dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Diagnose and generate a repair for a failure.

        Parameters
        ----------
        failure:
            Dictionary with keys ``traceback``, ``command``, ``exit_code``, etc.
        env:
            Current ``DockerEnvironment`` state.
        memory:
            Optional additional context (code_analysis, previous repairs).
        """
        llm_calls = 0
        traceback_text = failure.get("traceback", "")

        try:
            # ----- Step 1: normalise error ----------------------------
            signature = self._normalize_error(traceback_text)
            self.logger.info(
                "error_normalized",
                error_type=signature.error_type,
                hash=signature.hash,
            )

            # ----- Step 2: lookup known patterns ----------------------
            repair = self._lookup_skill(traceback_text, signature)
            if repair:
                self.logger.info("known_pattern_matched", desc=repair.description[:80])
                return AgentResult(
                    success=True,
                    data={
                        "repair": repair.model_dump(),
                        "signature": signature.model_dump(),
                        "source": "known_pattern",
                    },
                    llm_calls_used=0,
                )

            # ----- Step 3: check failure memory -----------------------
            prev_repair = self._retrieve_from_memory(signature)
            if prev_repair:
                self.logger.info("memory_hit", hash=signature.hash)
                return AgentResult(
                    success=True,
                    data={
                        "repair": prev_repair,
                        "signature": signature.model_dump(),
                        "source": "failure_memory",
                    },
                    llm_calls_used=0,
                )

            # ----- Step 4: LLM-generated repair -----------------------
            repair, calls = await self._generate_repair(failure, env, memory)
            llm_calls += calls

            # ----- Step 5: record in memory ---------------------------
            self._update_memory(signature, repair, success=None)

            return AgentResult(
                success=True,
                data={
                    "repair": repair.model_dump(),
                    "signature": signature.model_dump(),
                    "source": "llm_generated",
                },
                llm_calls_used=llm_calls,
            )

        except Exception as exc:
            self.logger.error("fix_agent_failed", error=str(exc))
            return AgentResult(
                success=False,
                error=str(exc),
                llm_calls_used=llm_calls,
            )

    # ------------------------------------------------------------------
    # Error normalisation
    # ------------------------------------------------------------------

    def _normalize_error(self, traceback_text: str) -> FailureSignature:
        """Extract a reusable failure signature from a traceback."""
        error_type = ""
        error_message = ""
        file_path = ""
        line_number = None

        lines = traceback_text.strip().splitlines()

        # Last line typically has the error type and message
        if lines:
            last_line = lines[-1].strip()
            if ":" in last_line:
                parts = last_line.split(":", 1)
                error_type = parts[0].strip()
                error_message = parts[1].strip() if len(parts) > 1 else ""

        # Find the deepest file reference
        file_pattern = re.compile(r'File "([^"]+)", line (\d+)')
        for line in reversed(lines):
            match = file_pattern.search(line)
            if match:
                file_path = match.group(1)
                line_number = int(match.group(2))
                break

        # Create a hash for deduplication
        hash_input = f"{error_type}:{error_message}".encode()
        sig_hash = hashlib.sha256(hash_input).hexdigest()[:12]

        return FailureSignature(
            error_type=error_type,
            error_message=error_message,
            file_path=file_path,
            line_number=line_number,
            hash=sig_hash,
        )

    # ------------------------------------------------------------------
    # Known pattern matching
    # ------------------------------------------------------------------

    def _lookup_skill(
        self, traceback_text: str, signature: FailureSignature
    ) -> RepairAction | None:
        """Match the error against known patterns (no LLM needed)."""
        for kp in KNOWN_PATTERNS:
            match = re.search(kp.pattern, traceback_text)
            if match:
                # Substitute match groups into fix template
                fix_text = kp.fix_template
                for i, group in enumerate(match.groups(), 1):
                    fix_text = fix_text.replace(f"{{match_{i}}}", group or "")
                    fix_text = fix_text.replace("{file}", signature.file_path)

                return RepairAction(
                    description=f"Known fix ({kp.category}): {fix_text}",
                    commands=[fix_text] if not fix_text.startswith("Check") else [],
                    confidence=0.8,
                )
        return None

    # ------------------------------------------------------------------
    # Failure memory
    # ------------------------------------------------------------------

    def _retrieve_from_memory(self, signature: FailureSignature) -> dict[str, Any] | None:
        """Check failure memory for a previous encounter of this signature."""
        if not self._memory_path.exists():
            return None

        try:
            data = json.loads(self._memory_path.read_text())
            return data.get(signature.hash)
        except (json.JSONDecodeError, IOError):
            return None

    def _update_memory(
        self,
        signature: FailureSignature,
        repair: RepairAction,
        success: bool | None,
    ) -> None:
        """Record a failure + repair in persistent memory."""
        self._memory_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {}
        if self._memory_path.exists():
            try:
                data = json.loads(self._memory_path.read_text())
            except (json.JSONDecodeError, IOError):
                data = {}

        data[signature.hash] = {
            "signature": signature.model_dump(),
            "repair": repair.model_dump(),
            "success": success,
        }

        self._memory_path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # LLM-generated repair
    # ------------------------------------------------------------------

    async def _generate_repair(
        self,
        failure: dict[str, Any],
        env: dict[str, Any] | None,
        memory: dict[str, Any] | None,
    ) -> tuple[RepairAction, int]:
        """Generate a repair via LLM when no known pattern matches."""
        traceback_text = failure.get("traceback", "")
        command = failure.get("command", "")

        context_parts = [
            f"Failed command: {command}",
            f"Exit code: {failure.get('exit_code', 'unknown')}",
            f"Traceback:\n{traceback_text[:3000]}",
        ]

        if env:
            context_parts.append(f"Docker image: {env.get('image_name', 'unknown')}")

        if memory:
            code_analysis = memory.get("code_analysis", "")
            if code_analysis:
                context_parts.append(
                    f"Code analysis (excerpt):\n{code_analysis[:1000]}"
                )

        messages = [
            {
                "role": "user",
                "content": (
                    "Diagnose this failure and provide a concrete repair.\n\n"
                    + "\n\n".join(context_parts)
                    + "\n\nProvide:\n"
                    "1. Brief description of the root cause\n"
                    "2. Shell commands to fix the issue\n"
                    "3. Any code changes needed (file, old_text, new_text)\n"
                    "4. Confidence level (0-1)"
                ),
            }
        ]
        result = await self._call_llm(messages, response_model=RepairAction)
        return result, 1
