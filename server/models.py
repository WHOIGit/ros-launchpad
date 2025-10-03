"""
Data models and enums for ROS Launchpad
"""

from datetime import datetime
from enum import Enum
from typing import Literal, Optional, TypedDict

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
    errors: list[str] = []
    warnings: list[str] = []


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
