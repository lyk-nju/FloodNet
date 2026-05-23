#!/bin/bash
# Repo-local pytest wrapper.
#
# Why this exists: this machine has ROS jazzy installed (/opt/ros/jazzy) which
# registers several pytest entrypoint plugins (launch_testing_ros, ament_*).
# Those target Python 3.12 site-packages and need `lark`, which our
# floodiffusion conda env (Python 3.10) doesn't have. The plugins load BEFORE
# pytest reads pytest.ini's `addopts`, so `-p no:<plugin>` flags via addopts
# come too late. The only reliable way to disable them is the env var
# PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, baked in here so callers (e.g. loop agents)
# don't need to remember it.
#
# Usage:
#   ./scripts/run_pytest.sh tests/test_foo.py -v --tb=short
set -euo pipefail
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec /home/lai/anaconda3/envs/floodiffusion/bin/python -m pytest "$@"
