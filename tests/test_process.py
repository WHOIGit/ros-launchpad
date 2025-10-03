"""
Unit tests for ROSProcess class
"""

import os
import pytest
from unittest.mock import Mock, patch
from server.process import ROSPRocess
from server.models import ProcessState


@pytest.fixture
def mock_env():
    """Fixture providing a basic environment"""
    return os.environ.copy()


def test_process_initialization(mock_env):
    """Test ROSProcess initialization"""
    process = ROSPRocess(
        name="test_process",
        command="echo hello",
        env=mock_env
    )
    assert process.name == "test_process"
    assert process.command == "echo hello"
    assert process.env == mock_env
    assert process.dont_kill is False
    assert process.required is True
    assert process.process is None
    assert process.info.state == ProcessState.STOPPED


def test_process_with_dont_kill(mock_env):
    """Test ROSProcess with dont_kill flag"""
    process = ROSPRocess(
        name="rosbag",
        command="rosbag record",
        env=mock_env,
        dont_kill=True
    )
    assert process.dont_kill is True


def test_is_running_when_not_started(mock_env):
    """Test is_running returns False when process not started"""
    process = ROSPRocess(
        name="test",
        command="echo hello",
        env=mock_env
    )
    assert process.is_running() is False


def test_get_info(mock_env):
    """Test get_info returns ProcessInfo"""
    process = ROSPRocess(
        name="test",
        command="echo hello",
        env=mock_env
    )
    info = process.get_info()
    assert info.name == "test"
    assert info.state == ProcessState.STOPPED
