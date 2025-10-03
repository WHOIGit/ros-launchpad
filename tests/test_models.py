"""
Unit tests for models
"""

import pytest
from datetime import datetime
from server.models import ProcessState, ProcessInfo, ConfigValidationResult


def test_process_state_enum():
    """Test ProcessState enum values"""
    assert ProcessState.STOPPED.value == "stopped"
    assert ProcessState.STARTING.value == "starting"
    assert ProcessState.RUNNING.value == "running"
    assert ProcessState.STOPPING.value == "stopping"
    assert ProcessState.FAILED.value == "failed"


def test_process_info_creation():
    """Test ProcessInfo model creation"""
    info = ProcessInfo(name="test_process", state=ProcessState.STOPPED)
    assert info.name == "test_process"
    assert info.state == ProcessState.STOPPED
    assert info.pid is None
    assert info.started_at is None
    assert info.stopped_at is None
    assert info.exit_code is None
    assert info.restart_count == 0


def test_process_info_with_all_fields():
    """Test ProcessInfo with all fields populated"""
    now = datetime.now()
    info = ProcessInfo(
        name="test_process",
        state=ProcessState.RUNNING,
        pid=1234,
        started_at=now,
        restart_count=5
    )
    assert info.name == "test_process"
    assert info.state == ProcessState.RUNNING
    assert info.pid == 1234
    assert info.started_at == now
    assert info.restart_count == 5


def test_config_validation_result_valid():
    """Test ConfigValidationResult for valid config"""
    result = ConfigValidationResult(valid=True)
    assert result.valid is True
    assert result.errors == []
    assert result.warnings == []


def test_config_validation_result_with_errors():
    """Test ConfigValidationResult with errors"""
    result = ConfigValidationResult(
        valid=False,
        errors=["Error 1", "Error 2"],
        warnings=["Warning 1"]
    )
    assert result.valid is False
    assert len(result.errors) == 2
    assert len(result.warnings) == 1
