export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export WANDB_BASE_URL=http://localhost:8080
export WANDB_API_KEY=local-2d50ec30d78b57814752748b4af4579ba81b4375
export DWAVE_API_TOKEN=DEV-39b2f9b4c4866b66903ccb7d11fa1d1166b1708e
export WANDB_DIR="/fast_scratch_1/caloqvae/luian1"
python $1
