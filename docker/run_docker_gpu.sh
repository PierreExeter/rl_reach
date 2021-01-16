#!/bin/bash
# Launch an experiment using the docker gpu image

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line

# TODO: always use new-style once sufficiently widely used (probably 2021 onwards)
if [ -x "$(which nvidia-docker)" ]; then
  # old-style nvidia-docker2
  NVIDIA_ARG="--runtime=nvidia"
else
  NVIDIA_ARG="--gpus all"
fi

docker run -it --rm --network host --ipc=host --mount src=$(pwd),target=/root/rl_reach/,type=bind rlreach/rlreach-gpu:latest bash -c "$cmd_line"