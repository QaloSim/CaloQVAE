export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export WANDB_BASE_URL=http://localhost:[port number]
export WANDB_API_KEY=[wandb key]
export DWAVE_API_TOKEN=[dwave api key]
python scripts/run.py
