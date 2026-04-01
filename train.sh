# use "setproxy" in tmux first

source /opt/anaconda3/bin/activate

conda activate flooddiffusion

# export HF_ENDPOINT="https://hf-mirror.com"

python train_ldf.py --config configs/ldf.yaml