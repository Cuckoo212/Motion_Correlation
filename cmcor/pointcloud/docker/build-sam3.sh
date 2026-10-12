#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
sudo -v
if ! command -v nvidia-ctk >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y curl gnupg2
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker
# Restarting Docker may interrupt running containers. Run while capture is stopped.
sudo systemctl restart docker
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)
sudo --preserve-env=LOCAL_UID,LOCAL_GID docker compose build gpu
sudo --preserve-env=LOCAL_UID,LOCAL_GID docker compose run --rm --no-deps gpu \
    /opt/sam3-env/bin/python -c 'import sam3, torch, rclpy; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0)); x=torch.randn(256,256,device="cuda"); y=x@x; torch.cuda.synchronize(); print("SAM3 import and CUDA OK")'
