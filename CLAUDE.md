# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ROS Launchpad is a web-based process management interface for ROS (Robot Operating System) applications. It provides a modern dashboard for starting/stopping ROS processes, monitoring logs in real-time, and managing configurations through a web interface built with FastAPI, HTMX, and Tailwind CSS.

## Development Setup

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

**Basic usage (no config file, with package name):**
```bash
python3 server.py --package my_robot_pkg
```

**With config file:**
```bash
python3 server.py path/to/config.yaml --package my_robot_pkg
```

**With direct launch directory path:**
```bash
python3 server.py --launch-dir /path/to/launch/files
```

**With auto-start processes:**
```bash
python3 server.py path/to/config.yaml --package my_robot_pkg --start roscore,main,rosbag
```

**Custom host/port:**
```bash
python3 server.py --package my_robot_pkg --host 0.0.0.0 --port 8080
```

**Using environment variables:**
```bash
export ROS_YAML_CONFIG=path/to/config.yaml
export LAUNCHPAD_AUTO_START=roscore,main
export LAUNCHPAD_PACKAGE=my_robot_pkg
# OR use direct launch directory:
export LAUNCHPAD_LAUNCH_DIR=/path/to/launch/files
python3 server.py
```

The server will be available at `http://localhost:8080` by default.

## Architecture

### Core Components

**server.py** - FastAPI application entry point
- Defines HTTP and WebSocket endpoints
- Handles dashboard rendering with HTMX
- Manages static file serving for Tailwind CSS and HTMX libraries
- Routes requests to LaunchpadServer

**server/dashboard.py** - LaunchpadServer class (main orchestration)
- Manages all ROS processes through ROSPRocess instances
- Handles config loading from files or URLs
- Discovers launch files at runtime from configurable package or directory
- Manages WebSocket connections for real-time log streaming
- Coordinates ROS parameter server updates via `rosparam` CLI
- Handles auto-start logic for processes

**server/process.py** - ROSPRocess class (individual process management)
- Manages lifecycle of a single process (roscore, rosbag, or launch file)
- Uses subprocess.Popen for process control
- Tracks process state, PID, uptime, restart counts
- Handles graceful termination vs. kill (configurable per-process with `dont_kill` flag)

**server/models.py** - Data models
- ProcessState enum: STOPPED, STARTING, RUNNING, STOPPING, FAILED
- ProcessInfo: tracks process metadata (PID, timestamps, exit codes)
- ConfigValidationResult: validation results for YAML configs

### Process Types

The system manages three categories of processes:

1. **Core processes**: `roscore` (ROS master node)
2. **Logging processes**: `rosbag` (ROS data recording)
3. **Launch configurations**: Discovered from launch files in the configured package or directory

### Configuration System

- YAML configuration files define ROS parameters and launch file settings
- Config validation happens via `scripts.config_validation.validate_config()`
- Configs can be loaded at startup, via web UI, or from URLs
- When config changes, parameters are pushed to ROS parameter server using `rosparam load`
- Server auto-detects launch files and creates process entries dynamically

### Real-time Features

**WebSocket log streaming** (`/ws/logs`):
- Clients connect and select log files to monitor
- Server streams new log lines as they're written
- Multiple clients can monitor different files simultaneously

**HTMX-driven UI updates**:
- Process cards refresh via partial HTML responses
- No page reloads needed for start/stop operations
- State transitions (starting → running) reflected in real-time

### Key Architectural Patterns

1. **Process Registry**: `LaunchpadServer.processes` dict maps process names to ROSPRocess instances
2. **Launch Discovery**: Launch files auto-discovered at runtime and added to available processes
3. **Environment Preparation**: `_prep_environment()` creates environment vars from config
4. **State Synchronization**: Background monitor task continuously checks process health
5. **Async Coordination**: Roscore startup waits for readiness before applying parameters

## Environment Variables

- `ROS_YAML_CONFIG`: Path to YAML configuration file (optional)
- `LAUNCHPAD_AUTO_START`: Comma-separated list of processes to auto-start (optional)
- `LAUNCHPAD_PACKAGE`: ROS package name for launch file discovery (optional, but recommended)
- `LAUNCHPAD_LAUNCH_DIR`: Direct path to launch files directory (optional, alternative to package name)

## Important Notes

- **ROS Dependency**: This application expects ROS to be installed and available (`roscore`, `roslaunch`, `rosparam` commands)
- **Launch File Discovery**: The system discovers launch files from either:
  - A specified ROS package: `src/<package_name>/launch/` (via `--package` or `LAUNCHPAD_PACKAGE`)
  - A direct directory path (via `--launch-dir` or `LAUNCHPAD_LAUNCH_DIR`)
  - If neither is provided, only `roscore` will be available (no launch file processes)
- **Config Validation**: References `scripts.config_validation` module (not present in current directory structure - may need to be added)
- **Rosbag Handling**: Uses `dont_kill=True` flag to allow graceful termination without SIGKILL
- **State Transitions**: UI buttons disabled during STARTING/STOPPING states to prevent race conditions
