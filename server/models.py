"""
Data models and enums for ROS Launchpad
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict


class ProcessState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


class ProcessInfo(BaseModel):
    name: str
    state: ProcessState
    pid: Optional[int] = None
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    restart_count: int = 0


class ConfigValidationResult(BaseModel):
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []


class ProcessMetadata(TypedDict, total=False):
    """Metadata for a process including description, type, and category"""
    description: str
    type: Literal["system", "launch", "unknown"]
    category: Literal["core", "logging", "mission", "simulation", "sensors", "other"]
    filename: str
    path: str


class ProcessData(TypedDict):
    """Process data structure used in rendering"""
    info: ProcessInfo
    metadata: ProcessMetadata
    priority: int


class ROSStatus(TypedDict):
    """ROS connectivity status"""
    ready: bool
    message: str


class ApiStatusResponse(TypedDict):
    """Response structure for /api/status endpoint"""
    processes: Dict[str, dict]  # ProcessInfo serialized to dict
    config_loaded: bool
    ros_status: ROSStatus


class LaunchConfig(TypedDict):
    """Launch configuration metadata"""
    filename: str
    description: str
    path: str


class LogFileInfo(TypedDict):
    """Log file metadata"""
    name: str
    path: str
    size: int
    modified: float


class LogFilesResponse(TypedDict, total=False):
    """Response structure for log files API"""
    log_dir: str
    files: List[LogFileInfo]
    error: str


class LogContentResponse(TypedDict, total=False):
    """Response structure for log file content API"""
    filename: str
    lines: List[str]
    total_lines: int
    file_size: int
    error: str


class AlertTestResult(TypedDict):
    """Result of alert system test"""
    success: bool
    message: str
