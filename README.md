# Wordle-GRPO

Here is a step by step instruction to get started on a new Virtual Machine using Ubuntu 22.4, Pytorch 12.8

apt-get update
<!-- Libenchant C compiler is required for Pyenchant -->
apt install libenchant-2-dev -y
<!-- Bashtop, nvtop and tmux to support monitoring the system performance -->
apt install bashtop -y
apt install nvtop -y
apt install tmux -y

<!-- Set up Git -->
apt-get install git gh -y
git config --global user.name "YPUR NAME"
git config --global user.email "YOUR EMAIL"
ssh-keygen -t rsa -b 4096 -C "YOUR EMAIL"

<!-- Make sure to create the SSH key in Github -->
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub

<!-- Copy over and paste the SSH key -->
ssh -T git@github.com

<!-- Clone the repository -->
git clone git@github.com:RLDiary/Wordle-GRPO.git
cd Wordle-GRPO

<!-- Install Rust compiler and UV -->
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

<!-- Initialise UV and install all packages; Note how flash-attn is installed here -->
uv venv --python 3.12.7 --seed
source .venv/bin/activate
uv sync --no-install-package flash-attn
uv sync

<!-- Initialise weights and biases and huggingface cli -->
wandb init
huggingface-cli login

<!-- Download the SFT model from Huggingface -->
cd ..
mkdir Models
cd Models
mkdir Qwen2.5-7B-WORDLE-FineTune
cd Qwen2.5-7B-WORDLE-FineTune
huggingface-cli download vigneshR/Qwen2.5-7B-WORDLE-FineTune --local-dir .
cd ..
cd ..
cd Wordle-GRPO

<!-- Create a new TMUX sessions -->
tmux new-session -n 0

<!-- Activate venv and run the code -->
source .venv/bin/activate
accelerate launch --config-file config/zero3.yaml --num-processes 2 Wordle/wordle_run.py