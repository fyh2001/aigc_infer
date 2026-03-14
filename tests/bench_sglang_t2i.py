"""
Benchmark: SGLang DiffGenerator text-to-image inference latency.

Loads the model once, runs N consecutive image generations, and reports
per-iteration wall time as well as aggregate statistics (avg / min / max / std).

Usage:
  python tests/bench_sglang_t2i.py \
      --prompt "A cat astronaut on the moon" \
      --num-runs 5 --warmup 1

  # With CPU offload options
  python tests/bench_sglang_t2i.py \
      --prompt "A futuristic city" \
      --text-encoder-cpu-offload --num-runs 10
"""

import argparse
import logging
import math
import os
import time
from pathlib import Path

import torch

from sglang.multimodal_gen import DiffGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ASPECT_RATIOS = {
    "1:1":  (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3":  (1472, 1104),
    "3:4":  (1104, 1472),
    "3:2":  (1584, 1056),
    "2:3":  (1056, 1584),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang T2I inference latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen-Image-2512")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--aspect-ratio", type=str, default="1:1", choices=list(ASPECT_RATIOS.keys()),
                        help="Aspect ratio preset (ignored if --height/--width are set)")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)

    perf_group = parser.add_argument_group("Performance options")
    perf_group.add_argument("--num-gpus", type=int, default=1)
    perf_group.add_argument("--no-cpu-offload", action="store_true")
    perf_group.add_argument("--text-encoder-cpu-offload", action="store_true")
    perf_group.add_argument("--pin-cpu-memory", action="store_true")
    perf_group.add_argument("--enable-torch-compile", action="store_true",
                            help="Use torch.compile to speed up DiT inference")
    perf_group.add_argument("--attention-backend", type=str, default=None,
                            choices=["fa", "sage_attn", "torch_sdpa"],
                            help="Attention backend (default: auto-detect)")
    perf_group.add_argument("--warmup-engine", action="store_true",
                            help="Enable SGLang internal engine warmup for better kernel performance")

    bench_group = parser.add_argument_group("Benchmark parameters")
    bench_group.add_argument("--num-runs", type=int, default=5, help="Number of timed inference runs")
    bench_group.add_argument("--warmup", type=int, default=1, help="Number of warmup runs (excluded from stats)")
    bench_group.add_argument("--output-dir", type=str, default="outputs_bench_sglang/")
    bench_group.add_argument("--no-save", action="store_true", help="Skip saving images to disk")

    return parser.parse_args()


def resolve_dimensions(args):
    if args.height is not None and args.width is not None:
        return args.width, args.height
    w, h = ASPECT_RATIOS[args.aspect_ratio]
    return w, h


def build_sampling_kwargs(args, width, height, run_idx, save=True):
    sampling_kwargs = dict(
        prompt=args.prompt,
        output_path=args.output_dir,
        save_output=save,
        seed=args.seed + run_idx,
        num_outputs_per_prompt=1,
        height=height,
        width=width,
    )
    if args.negative_prompt is not None:
        sampling_kwargs["negative_prompt"] = args.negative_prompt
    if args.steps is not None:
        sampling_kwargs["num_inference_steps"] = args.steps
    if args.cfg_scale is not None:
        sampling_kwargs["guidance_scale"] = args.cfg_scale
    if save:
        sampling_kwargs["output_file_name"] = f"bench_{run_idx:04d}"
    return sampling_kwargs


def run_once(generator, args, width, height, run_idx, save=True):
    sampling_kwargs = build_sampling_kwargs(args, width, height, run_idx, save=save)

    t0 = time.perf_counter()
    result = generator.generate(sampling_params_kwargs=dict(sampling_kwargs))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return elapsed


def main():
    args = parse_args()
    width, height = resolve_dimensions(args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Benchmark: SGLang T2I (DiffGenerator)")
    logger.info("  Model:        %s", args.model_path)
    logger.info("  Prompt:       %s", args.prompt)
    logger.info("  Size:         %dx%d", width, height)
    logger.info("  Warmup:       %d", args.warmup)
    logger.info("  Num runs:     %d", args.num_runs)
    logger.info("  torch.compile: %s", args.enable_torch_compile)
    logger.info("  attn_backend: %s", args.attention_backend or "auto")
    logger.info("  engine_warmup: %s", args.warmup_engine)
    logger.info("  cache_dit:    %s", os.environ.get("SGLANG_CACHE_DIT_ENABLED", "false"))
    logger.info("=" * 70)

    # --- Load model ---
    server_kwargs = dict(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
    )
    if args.no_cpu_offload:
        server_kwargs["dit_cpu_offload"] = False
        server_kwargs["text_encoder_cpu_offload"] = False
        server_kwargs["vae_cpu_offload"] = False
        server_kwargs["pin_cpu_memory"] = False
    else:
        if args.text_encoder_cpu_offload:
            server_kwargs["text_encoder_cpu_offload"] = True
        if args.pin_cpu_memory:
            server_kwargs["pin_cpu_memory"] = True
    if args.lora_path:
        server_kwargs["lora_path"] = args.lora_path
    if args.enable_torch_compile:
        server_kwargs["enable_torch_compile"] = True
    if args.attention_backend:
        server_kwargs["attention_backend"] = args.attention_backend
    if args.warmup_engine:
        server_kwargs["warmup"] = True

    model_load_t0 = time.perf_counter()
    generator = DiffGenerator.from_pretrained(**server_kwargs)
    model_load_elapsed = time.perf_counter() - model_load_t0
    logger.info("Model loaded in %.3fs", model_load_elapsed)

    try:
        # --- Warmup ---
        for wi in range(args.warmup):
            t = run_once(generator, args, width, height, run_idx=wi, save=False)
            logger.info("[Warmup %d/%d] %.3fs", wi + 1, args.warmup, t)
        if args.warmup:
            logger.info("Warmup complete.\n")

        # --- Timed runs ---
        times = []
        save = not args.no_save
        for ri in range(args.num_runs):
            t = run_once(generator, args, width, height, run_idx=ri, save=save)
            times.append(t)
            logger.info("[Run %d/%d] %.3fs", ri + 1, args.num_runs, t)

        # --- Statistics ---
        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        std = math.sqrt(sum((t - avg) ** 2 for t in times) / len(times))

        logger.info("")
        logger.info("=" * 70)
        logger.info("Results  (%d runs, %d warmup)", args.num_runs, args.warmup)
        logger.info("-" * 70)
        logger.info("  avg   = %.3fs", avg)
        logger.info("  min   = %.3fs", mn)
        logger.info("  max   = %.3fs", mx)
        logger.info("  std   = %.3fs", std)
        logger.info("  model_load = %.3fs", model_load_elapsed)
        logger.info("-" * 70)
        for i, t in enumerate(times):
            logger.info("  run %d: %.3fs", i + 1, t)
        logger.info("=" * 70)

    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
