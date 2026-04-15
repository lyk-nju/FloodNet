# use "setproxy" in tmux first
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
# export HF_ENDPOINT="https://hf-mirror.com"

source /opt/anaconda3/bin/activate

conda activate flooddiffusion

python train_ldf.py --config configs/ldf.yaml