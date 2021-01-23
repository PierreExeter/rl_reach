#!/bin/bash
# Launch an experiment using the docker gpu image

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line

docker run -it --rm --runtime=nvidia --network host --ipc=host --mount src=$(pwd),target=/root/rl_reach/,type=bind rlreach/rlreach-gpu:latest bash -c "$cmd_line"

# OR

# docker run -it --rm --gpus all --network host --ipc=host --mount src=$(pwd),target=/root/rl_reach/,type=bind rlreach/rlreach-gpu:latest bash -c "$cmd_line"