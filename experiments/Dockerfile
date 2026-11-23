# Training image for DGX Spark (aarch64, GB10): reuses torch/transformers from the vLLM image
# that already works on this GPU, adds the libraries for LoRA, metrics and plots.
# Build: docker build -t crypti-train -f experiments/Dockerfile experiments
FROM vllm/vllm-openai:latest
RUN pip freeze | grep -iE "^(torch|transformers|triton|accelerate|numpy)==" > /tmp/pins.txt \
 && pip install --no-cache-dir -c /tmp/pins.txt peft scikit-learn pandas matplotlib scipy flash-linear-attention
# fast path of Qwen3.5 (Gated DeltaNet) in transformers needs causal-conv1d besides fla; no aarch64 wheel -> build
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 pip install --no-cache-dir --no-build-isolation causal-conv1d==1.7.0
ENTRYPOINT []
