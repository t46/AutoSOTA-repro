"""Docker container management for running paper experiments.

All docker-py operations are synchronous and are wrapped with
:func:`asyncio.to_thread` to provide an async interface.
"""

from __future__ import annotations

import asyncio
import io
import logging
import tarfile
import tempfile
from pathlib import Path

import docker
from docker.errors import DockerException, ImageNotFound, NotFound

logger = logging.getLogger(__name__)


class DockerError(Exception):
    """Raised when a Docker operation fails."""


class DockerManager:
    """Async wrapper around the Docker SDK for Python.

    Parameters
    ----------
    platform:
        Target platform for image builds/pulls (e.g. ``"linux/arm64"``).
    memory_limit:
        Memory limit for containers (e.g. ``"16g"``).
    network_mode:
        Docker network mode (``"bridge"``, ``"host"``, etc.).
    timeout:
        Default command execution timeout in seconds.
    """

    def __init__(
        self,
        platform: str = "linux/arm64",
        memory_limit: str = "16g",
        network_mode: str = "bridge",
        timeout: int = 7200,
    ) -> None:
        self.platform = platform
        self.memory_limit = memory_limit
        self.network_mode = network_mode
        self.timeout = timeout
        self._client: docker.DockerClient | None = None

    @property
    def client(self) -> docker.DockerClient:
        """Lazy-initialise and return the Docker client."""
        if self._client is None:
            try:
                self._client = docker.from_env()
            except DockerException as exc:
                raise DockerError(f"Cannot connect to Docker daemon: {exc}") from exc
        return self._client

    # ------------------------------------------------------------------
    # Image operations
    # ------------------------------------------------------------------

    async def build_image(
        self,
        dockerfile_path: Path | None,
        base_image: str,
        tag: str,
        build_commands: list[str] | None = None,
    ) -> str:
        """Build or pull a Docker image. Returns the image tag.

        If *dockerfile_path* is provided, builds from that Dockerfile.
        Otherwise pulls *base_image* and optionally runs *build_commands*
        inside a temporary container, committing the result with *tag*.
        """
        if dockerfile_path and dockerfile_path.is_file():
            return await self._build_from_dockerfile(dockerfile_path, tag)

        # Pull the base image.
        await self._pull_image(base_image)

        if not build_commands:
            # Just tag the pulled image.
            await asyncio.to_thread(self._tag_image, base_image, tag)
            return tag

        # Run build commands in a temporary container and commit.
        container_id = await self.create_container(
            image=base_image,
            name=f"build-{tag.replace(':', '-').replace('/', '-')}",
        )
        try:
            for cmd in build_commands:
                exit_code, output = await self.exec_command(container_id, cmd)
                if exit_code != 0:
                    raise DockerError(
                        f"Build command failed (exit {exit_code}): {cmd}\n{output}"
                    )
            await self.commit_container(container_id, tag)
        finally:
            await self.stop_and_remove(container_id)

        return tag

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    async def create_container(
        self,
        image: str,
        name: str,
        working_dir: str = "/workspace",
        env_vars: dict | None = None,
        gpu_runtime: str | None = None,
    ) -> str:
        """Create and start a container. Returns the container ID."""

        def _create() -> str:
            kwargs: dict = {
                "image": image,
                "name": name,
                "working_dir": working_dir,
                "detach": True,
                "tty": True,
                "stdin_open": True,
                "mem_limit": self.memory_limit,
                "network_mode": self.network_mode,
                "platform": self.platform,
            }
            if env_vars:
                kwargs["environment"] = env_vars
            if gpu_runtime:
                kwargs["runtime"] = gpu_runtime

            container = self.client.containers.create(**kwargs)
            container.start()
            return container.id

        return await asyncio.to_thread(_create)

    async def exec_command(
        self,
        container_id: str,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> tuple[int, str]:
        """Execute a command inside a running container.

        Returns ``(exit_code, combined_output)``.
        """
        effective_timeout = timeout or self.timeout

        def _exec() -> tuple[int, str]:
            container = self.client.containers.get(container_id)
            kwargs: dict = {"cmd": ["/bin/bash", "-c", command], "demux": True}
            if workdir:
                kwargs["workdir"] = workdir

            exec_result = container.exec_run(**kwargs)
            exit_code: int = exec_result.exit_code

            # demux=True returns (stdout, stderr) or plain bytes depending
            # on the docker SDK version / whether TTY is enabled.
            output_raw = exec_result.output
            if isinstance(output_raw, tuple):
                stdout = (output_raw[0] or b"").decode("utf-8", errors="replace")
                stderr = (output_raw[1] or b"").decode("utf-8", errors="replace")
                combined = stdout + stderr
            elif isinstance(output_raw, bytes):
                combined = output_raw.decode("utf-8", errors="replace")
            else:
                combined = str(output_raw)

            return exit_code, combined

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_exec), timeout=effective_timeout
            )
        except asyncio.TimeoutError:
            logger.error("Command timed out after %ds: %s", effective_timeout, command)
            return -1, f"Command timed out after {effective_timeout}s"

    # ------------------------------------------------------------------
    # File transfer
    # ------------------------------------------------------------------

    async def copy_to_container(
        self, container_id: str, src: Path, dest: str
    ) -> None:
        """Copy a local file or directory into the container at *dest*."""

        def _copy() -> None:
            container = self.client.containers.get(container_id)
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                tar.add(str(src), arcname=Path(src).name)
            buf.seek(0)
            container.put_archive(dest, buf)

        await asyncio.to_thread(_copy)

    async def copy_from_container(
        self, container_id: str, src: str, dest: Path
    ) -> None:
        """Copy a file or directory from the container at *src* to local *dest*."""

        def _copy() -> None:
            container = self.client.containers.get(container_id)
            bits, _ = container.get_archive(src)

            dest.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tar") as tmp:
                for chunk in bits:
                    tmp.write(chunk)
                tmp_path = tmp.name

            with tarfile.open(tmp_path) as tar:
                tar.extractall(path=str(dest.parent))

            Path(tmp_path).unlink(missing_ok=True)

        await asyncio.to_thread(_copy)

    # ------------------------------------------------------------------
    # Container management
    # ------------------------------------------------------------------

    async def commit_container(self, container_id: str, tag: str) -> str:
        """Commit a container's current state as a new image. Returns the image tag."""

        def _commit() -> str:
            container = self.client.containers.get(container_id)
            parts = tag.rsplit(":", 1)
            repo = parts[0]
            tag_name = parts[1] if len(parts) > 1 else "latest"
            container.commit(repository=repo, tag=tag_name)
            return tag

        return await asyncio.to_thread(_commit)

    async def stop_and_remove(self, container_id: str) -> None:
        """Stop and remove a container, ignoring errors if already gone."""

        def _stop_remove() -> None:
            try:
                container = self.client.containers.get(container_id)
                container.stop(timeout=10)
            except (NotFound, DockerException):
                pass
            try:
                container = self.client.containers.get(container_id)
                container.remove(force=True)
            except (NotFound, DockerException):
                pass

        await asyncio.to_thread(_stop_remove)

    async def container_exists(self, container_id: str) -> bool:
        """Check whether a container exists (running or stopped)."""

        def _exists() -> bool:
            try:
                self.client.containers.get(container_id)
                return True
            except NotFound:
                return False

        return await asyncio.to_thread(_exists)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _pull_image(self, image: str) -> None:
        """Pull an image from a registry."""

        def _pull() -> None:
            logger.info("Pulling image %s (platform=%s)", image, self.platform)
            self.client.images.pull(image, platform=self.platform)

        await asyncio.to_thread(_pull)

    async def _build_from_dockerfile(self, dockerfile_path: Path, tag: str) -> str:
        """Build an image from a Dockerfile."""

        def _build() -> str:
            logger.info("Building image from %s as %s", dockerfile_path, tag)
            self.client.images.build(
                path=str(dockerfile_path.parent),
                dockerfile=dockerfile_path.name,
                tag=tag,
                platform=self.platform,
                rm=True,
            )
            return tag

        return await asyncio.to_thread(_build)

    def _tag_image(self, source: str, target: str) -> None:
        """Tag an existing image."""
        try:
            image = self.client.images.get(source)
        except ImageNotFound:
            raise DockerError(f"Image not found: {source}")
        parts = target.rsplit(":", 1)
        repo = parts[0]
        tag_name = parts[1] if len(parts) > 1 else "latest"
        image.tag(repo, tag=tag_name)
