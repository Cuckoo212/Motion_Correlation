#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
source /opt/cable_ws/install/setup.bash
exec "$@"
