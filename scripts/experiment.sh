#!/usr/bin/env bash
# Execute, never source. Domain 113 is separate from normal sim 112 / real 42.
set -euo pipefail
experiment_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -eq 0 ]]; then
    echo 'Usage: bash scripts/experiment.sh COMMAND [ARG ...]' >&2
    exit 2
fi
experiment_gui=()
for experiment_key in DISPLAY XAUTHORITY XDG_RUNTIME_DIR WAYLAND_DISPLAY; do
    if [[ -n "${!experiment_key:-}" ]]; then
        experiment_gui+=("$experiment_key=${!experiment_key}")
    fi
done
exec env -i HOME="$HOME" USER="${USER:-sukja}" PATH=/usr/bin:/bin \
    LANG=C.UTF-8 ROS_DOMAIN_ID=113 ROS_LOCALHOST_ONLY=1 \
    ROS_LOG_DIR="$HOME/terrain_nav_data/experiments/ros_logs" \
    PYTHONDONTWRITEBYTECODE=1 "${experiment_gui[@]}" \
    bash --noprofile --norc -c '
        set -eo pipefail
        experiment_root=$1
        shift
        source /opt/ros/humble/setup.bash
        if [[ -f "$experiment_root/install/local_setup.bash" ]]; then
            source "$experiment_root/install/local_setup.bash"
        fi
        export PYTHONPATH="$experiment_root/src/terrain_navigation_pkg:${PYTHONPATH:-}"
        cd "$experiment_root"
        exec "$@"
    ' simulation-experiment "$experiment_root" "$@"
