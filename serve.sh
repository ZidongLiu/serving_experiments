export VLLM_USE_FLASHINFER_SAMPLER=0

uv run python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='facebook/opt-125m', gpu_memory_utilization=0.15, max_model_len=512)
print(llm.generate(['The capital of France is'], SamplingParams(max_tokens=8))[0].outputs[0].text)
"