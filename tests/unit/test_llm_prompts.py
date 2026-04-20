"""Unit tests for autosota.llm.prompts."""

from __future__ import annotations

import pytest

from autosota.llm.prompts import PromptManager


@pytest.fixture
def prompts_dir(tmp_path):
    """Create a temporary prompts directory with sample files."""
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "analyzer_system.md").write_text("You are the analyzer agent.\n")
    (d / "coder_system.md").write_text("You are the coder agent.\n")
    (d / "report.md").write_text("Report for {{ paper_title }} by {{ author }}.\n")
    return d


class TestGetSystemPrompt:
    def test_loads_existing_prompt(self, prompts_dir):
        pm = PromptManager(prompts_dir)
        text = pm.get_system_prompt("analyzer")
        assert "analyzer agent" in text

    def test_raises_on_missing_prompt(self, prompts_dir):
        pm = PromptManager(prompts_dir)
        with pytest.raises(FileNotFoundError, match="not_a_thing_system.md"):
            pm.get_system_prompt("not_a_thing")

    def test_caches_prompt(self, prompts_dir):
        pm = PromptManager(prompts_dir)
        text1 = pm.get_system_prompt("coder")
        text2 = pm.get_system_prompt("coder")
        assert text1 is text2  # Same object due to caching.


class TestRender:
    def test_renders_template(self, prompts_dir):
        pm = PromptManager(prompts_dir)
        result = pm.render("report.md", paper_title="DCSR", author="Alice")
        assert "DCSR" in result
        assert "Alice" in result

    def test_raises_on_missing_template(self, prompts_dir):
        pm = PromptManager(prompts_dir)
        with pytest.raises(FileNotFoundError, match="nonexistent.md"):
            pm.render("nonexistent.md")


class TestListTemplates:
    def test_lists_all_files(self, prompts_dir):
        pm = PromptManager(prompts_dir)
        templates = pm.list_templates()
        assert "analyzer_system.md" in templates
        assert "coder_system.md" in templates
        assert "report.md" in templates

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        pm = PromptManager(d)
        assert pm.list_templates() == []

    def test_nonexistent_dir(self, tmp_path):
        pm = PromptManager(tmp_path / "nope")
        assert pm.list_templates() == []
