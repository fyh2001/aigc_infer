#!/usr/bin/env bash
# ============================================================================
# Profile commands for comparing different optimization configurations.
#
# SGLang: uses built-in --profile (torch.profiler inside worker process)
#   Output: .trace.json.gz -> open in https://ui.perfetto.dev
#
# Diffusers: uses --profile (torch.profiler in same process)
#   Output: .json -> open in https://ui.perfetto.dev
#
# Warmup strategy (--profile-warmup N):
#   Warmup N 次不带 profiling 的生成，预热 CUDA kernels / 内存分配，
#   然后再跑 1 次带 profiling 的生成，trace 只包含稳态性能数据。
#   - 普通命令: warmup=1 (预热 CUDA kernel JIT)
#   - torch.compile: warmup=2 (第1次触发编译，第2次消化 shape specialization)
#
# Run commands ONE AT A TIME, not the whole script at once.
# ============================================================================

PROMPT="A cat astronaut on the moon"
IMAGE_PATH="input.jpg"  # <-- replace with your test image for edit scripts
STEPS=40

# ============================================================================
# 1. SGLang T2I
# ============================================================================

# 1a. SGLang T2I — baseline (FlashAttention)
#   Trace output: ./logs/profile_trace-full_stages-global-rank0.trace.json.gz
#   IMPORTANT: rename/move the trace file before running the next command!
uv run scripts/run_qwen_image_t2i.py \
    --prompt "$PROMPT" --steps $STEPS \
    --no-cpu-offload \
    --profile --profile-all-stages --profile-warmup 1
mv ./logs/profile_trace-full_stages-global-rank0.trace.json.gz ./profiler_traces/sglang_t2i_baseline.trace.json.gz

# 1b. SGLang T2I — TeaCache (default aggressive)
SGLANG_CACHE_DIT_ENABLED=true \
uv run scripts/run_qwen_image_t2i.py \
    --prompt "$PROMPT" --steps $STEPS \
    --no-cpu-offload \
    --profile --profile-all-stages --profile-warmup 1
mv ./logs/profile_trace-full_stages-global-rank0.trace.json.gz ./profiler_traces/sglang_t2i_teacache_default.trace.json.gz

# 1c. SGLang T2I — TeaCache (conservative + TaylorSeer)
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_RDT=0.10 \
SGLANG_CACHE_DIT_MC=2 \
SGLANG_CACHE_DIT_FN=2 \
SGLANG_CACHE_DIT_BN=1 \
SGLANG_CACHE_DIT_TAYLORSEER=true \
SGLANG_CACHE_DIT_TS_ORDER=1 \
uv run scripts/run_qwen_image_t2i.py \
    --prompt "$PROMPT" --steps $STEPS \
    --no-cpu-offload \
    --profile --profile-all-stages --profile-warmup 1
mv ./logs/profile_trace-full_stages-global-rank0.trace.json.gz ./profiler_traces/sglang_t2i_teacache_conservative.trace.json.gz

# 1d. SGLang T2I — torch.compile
uv run scripts/run_qwen_image_t2i.py \
    --prompt "$PROMPT" --steps $STEPS \
    --no-cpu-offload \
    --enable-torch-compile \
    --profile --profile-all-stages --profile-warmup 2
mv ./logs/profile_trace-full_stages-global-rank0.trace.json.gz ./profiler_traces/sglang_t2i_compile.trace.json.gz

# 1e. SGLang T2I — SageAttention
uv run scripts/run_qwen_image_t2i.py \
    --prompt "$PROMPT" --steps $STEPS \
    --no-cpu-offload \
    --attention-backend sage_attn \
    --profile --profile-all-stages --profile-warmup 1
mv ./logs/profile_trace-full_stages-global-rank0.trace.json.gz ./profiler_traces/sglang_t2i_sage_attn.trace.json.gz

# 1f. SGLang T2I — 4 steps
uv run scripts/run_qwen_image_t2i.py \
    --prompt "$PROMPT" --steps 4 \
    --no-cpu-offload \
    --profile --profile-all-stages --profile-warmup 1
mv ./logs/profile_trace-full_stages-global-rank0.trace.json.gz ./profiler_traces/sglang_t2i_4steps.trace.json.gz

# ============================================================================
# 2. Diffusers T2I
# ============================================================================

2a. Diffusers T2I — baseline
uv run scripts/run_diffusers_image_t2i.py \
    --prompt "$PROMPT" --steps $STEPS \
    --profile --profile-warmup 1 --profile-repeat 1 \
    --profile-summary \
    --profile-with-stack \
    --profile-memory \
    --profile-dir ./profiler_traces/diffusers_t2i_baseline

# 2b. Diffusers T2I — torch.compile
python scripts/run_diffusers_image_t2i.py \
    --prompt "$PROMPT" --steps $STEPS \
    --torch-compile \
    --profile --profile-warmup 2 --profile-repeat 1 \
    --profile-summary \
    --profile-with-stack \
    --profile-memory \ 
    --profile-dir ./profiler_traces/diffusers_t2i_compile

# 2c. Diffusers T2I — 4 steps
python scripts/run_diffusers_image_t2i.py \
    --prompt "$PROMPT" --steps 4 \
    --profile --profile-warmup 1 --profile-repeat 1 \
    --profile-summary \
    --profile-dir ./profiler_traces/diffusers_t2i_4steps

# ============================================================================
# 3. SGLang Edit (uncomment and set IMAGE_PATH)
# ============================================================================

# # 3a. SGLang Edit — baseline
# uv run scripts/run_qwen_image_edit.py \
#     --image-path "$IMAGE_PATH" \
#     --prompt "Turn it into a watercolor painting." --steps $STEPS \
#     --no-cpu-offload \
#     --profile --profile-all-stages --profile-warmup 1
# mv ./logs/profile_trace-full_stages-global-rank0.trace.json.gz ./profiler_traces/sglang_edit_baseline.trace.json.gz

# # 3b. SGLang Edit — TeaCache
# SGLANG_CACHE_DIT_ENABLED=true \
# uv run scripts/run_qwen_image_edit.py \
#     --image-path "$IMAGE_PATH" \
#     --prompt "Turn it into a watercolor painting." --steps $STEPS \
#     --no-cpu-offload \
#     --profile --profile-all-stages --profile-warmup 1
# mv ./logs/profile_trace-full_stages-global-rank0.trace.json.gz ./profiler_traces/sglang_edit_teacache.trace.json.gz

# ============================================================================
# 4. Diffusers Edit (uncomment and set IMAGE_PATH)
# ============================================================================

# # 4a. Diffusers Edit — baseline
# python scripts/run_diffusers_image_edit.py \
#     --image-path "$IMAGE_PATH" \
#     --prompt "Turn it into a watercolor painting." --steps $STEPS \
#     --profile --profile-warmup 1 --profile-repeat 1 \
#     --profile-summary \
#     --profile-dir ./profiler_traces/diffusers_edit_baseline

# # 4b. Diffusers Edit — torch.compile
# python scripts/run_diffusers_image_edit.py \
#     --image-path "$IMAGE_PATH" \
#     --prompt "Turn it into a watercolor painting." --steps $STEPS \
#     --torch-compile \
#     --profile --profile-warmup 2 --profile-repeat 1 \
#     --profile-summary \
#     --profile-dir ./profiler_traces/diffusers_edit_compile
