#!/usr/bin/env bash
# Run inside the GPU container. Logs survive container removal through /workspace.
set -eo pipefail
source /opt/ros/humble/setup.bash
source /opt/cmcor/install/setup.bash
set -u
export PYTHONPATH="/workspace/cmcor/pointcloud:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
command -v gdb >/dev/null || { echo "Install gdb in the diagnostic container first" >&2; exit 1; }
log_dir=/workspace/cmcor/pointcloud/diagnostics
mkdir -p "$log_dir"
log_file="$log_dir/gdb_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "Saving diagnostic output to $log_file"
gdb -batch \
    -ex 'set pagination off' \
    -ex 'set print thread-events off' \
    -ex run \
    -ex 'x/12i $pc' \
    -ex 'info registers' \
    -ex 'info proc mappings' \
    -ex 'thread apply all bt 30' \
    --args /opt/sam3-env/bin/python \
    /opt/cmcor/install/cable_pointcloud/lib/cable_pointcloud/sam3_pointcloud \
    --no-view "$@" 2>&1 | tee "$log_file"
