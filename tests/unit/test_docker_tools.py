"""Unit tests for autosota.tools.docker_tools.

These tests mock the Docker client to avoid requiring a running Docker daemon.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autosota.tools.docker_tools import DockerError, DockerManager


@pytest.fixture
def docker_mgr():
    """Return a DockerManager with a mocked client."""
    mgr = DockerManager(platform="linux/arm64", memory_limit="8g")
    mgr._client = MagicMock()
    return mgr


class TestDockerManagerInit:
    def test_defaults(self):
        mgr = DockerManager()
        assert mgr.platform == "linux/arm64"
        assert mgr.memory_limit == "16g"
        assert mgr.network_mode == "bridge"
        assert mgr.timeout == 7200

    def test_custom_settings(self):
        mgr = DockerManager(platform="linux/amd64", memory_limit="4g", timeout=600)
        assert mgr.platform == "linux/amd64"
        assert mgr.memory_limit == "4g"


class TestCreateContainer:
    async def test_create_and_start(self, docker_mgr):
        mock_container = MagicMock()
        mock_container.id = "abc123"
        docker_mgr._client.containers.create.return_value = mock_container

        cid = await docker_mgr.create_container(
            image="python:3.10", name="test-container"
        )
        assert cid == "abc123"
        mock_container.start.assert_called_once()

    async def test_env_vars_passed(self, docker_mgr):
        mock_container = MagicMock()
        mock_container.id = "abc123"
        docker_mgr._client.containers.create.return_value = mock_container

        await docker_mgr.create_container(
            image="python:3.10",
            name="test",
            env_vars={"KEY": "VALUE"},
        )
        call_kwargs = docker_mgr._client.containers.create.call_args[1]
        assert call_kwargs["environment"] == {"KEY": "VALUE"}


class TestExecCommand:
    async def test_exec_returns_output(self, docker_mgr):
        mock_container = MagicMock()
        mock_exec_result = MagicMock()
        mock_exec_result.exit_code = 0
        mock_exec_result.output = (b"hello\n", b"")
        mock_container.exec_run.return_value = mock_exec_result
        docker_mgr._client.containers.get.return_value = mock_container

        exit_code, output = await docker_mgr.exec_command("abc123", "echo hello")
        assert exit_code == 0
        assert "hello" in output

    async def test_exec_nonzero_exit(self, docker_mgr):
        mock_container = MagicMock()
        mock_exec_result = MagicMock()
        mock_exec_result.exit_code = 1
        mock_exec_result.output = (b"", b"error msg\n")
        mock_container.exec_run.return_value = mock_exec_result
        docker_mgr._client.containers.get.return_value = mock_container

        exit_code, output = await docker_mgr.exec_command("abc123", "bad_cmd")
        assert exit_code == 1
        assert "error msg" in output


class TestContainerExists:
    async def test_exists(self, docker_mgr):
        docker_mgr._client.containers.get.return_value = MagicMock()
        assert await docker_mgr.container_exists("abc123") is True

    async def test_not_exists(self, docker_mgr):
        from docker.errors import NotFound

        docker_mgr._client.containers.get.side_effect = NotFound("gone")
        assert await docker_mgr.container_exists("xyz") is False


class TestStopAndRemove:
    async def test_stop_and_remove(self, docker_mgr):
        mock_container = MagicMock()
        docker_mgr._client.containers.get.return_value = mock_container

        await docker_mgr.stop_and_remove("abc123")
        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()


class TestCommitContainer:
    async def test_commit(self, docker_mgr):
        mock_container = MagicMock()
        docker_mgr._client.containers.get.return_value = mock_container

        tag = await docker_mgr.commit_container("abc123", "myimage:v1")
        assert tag == "myimage:v1"
        mock_container.commit.assert_called_once_with(repository="myimage", tag="v1")
