export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --max-num-seqs 128 \
  --max-num-batched-tokens 2048