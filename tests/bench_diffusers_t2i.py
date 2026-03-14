"""
Benchmark: Diffusers pipeline text-to-image inference latency.

Loads the model once, runs N consecutive image generations, and reports
per-iteration wall time as well as aggregate statistics (avg / min / max / std).

Usage:
  python tests/bench_diffusers_t2i.py \
      --prompt "A cat astronaut on the moon" \
      --num-runs 5 --warmup 1

  # Custom resolution
  python tests/bench_diffusers_t2i.py \
      --prompt "A futuristic city" --aspect-ratio 16:9 --num-runs 10
"""

import argparse
import logging
import math
import time
from pathlib import Path

import torch

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
        description="Benchmark Diffusers T2I inference latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen-Image-2512")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--negative-prompt", type=str,
        default="低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
    )
    parser.add_argument("--aspect-ratio", type=str, default="1:1", choices=list(ASPECT_RATIOS.keys()))
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--device", type=str, default=None)

    bench_group = parser.add_argument_group("Benchmark parameters")
    bench_group.add_argument("--num-runs", type=int, default=5, help="Number of timed inference runs")
    bench_group.add_argument("--warmup", type=int, default=1, help="Number of warmup runs (excluded from stats)")
    bench_group.add_argument("--output-dir", type=str, default="outputs_bench_diffusers/")
    bench_group.add_argument("--no-save", action="store_true", help="Skip saving images to disk")

    return parser.parse_args()


def resolve_device(args):
    if args.device:
        return args.device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dimensions(args):
    if args.height is not None and args.width is not None:
        return args.width, args.height
    w, h = ASPECT_RATIOS[args.aspect_ratio]
    return w, h


def load_pipeline(args, device):
    from diffusers import DiffusionPipeline

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    logger.info("Loading pipeline  model=%s  dtype=%s  device=%s", args.model_path, dtype, device)

    pipeline = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=dtype)

    if args.cpu_offload and device == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)

    return pipeline


def run_once(pipeline, args, device, width, height, run_idx, save=True):
    gen_device = device if device == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(args.seed + run_idx)

    t0 = time.perf_counter()
    with torch.inference_mode():
        output = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=args.steps,
            true_cfg_scale=args.cfg_scale,
            num_images_per_prompt=1,
            generator=generator,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"bench_{run_idx:04d}.png"
        output.images[0].save(str(output_dir / fname))

    return elapsed


def main():
    args = parse_args()
    device = resolve_device(args)
    width, height = resolve_dimensions(args)

    logger.info("=" * 70)
    logger.info("Benchmark: Diffusers T2I")
    logger.info("  Model:      %s", args.model_path)
    logger.info("  Prompt:     %s", args.prompt)
    logger.info("  Size:       %dx%d", width, height)
    logger.info("  Steps:      %d", args.steps)
    logger.info("  Device:     %s", device)
    logger.info("  Warmup:     %d", args.warmup)
    logger.info("  Num runs:   %d", args.num_runs)
    logger.info("=" * 70)

    model_load_t0 = time.perf_counter()
    pipeline = load_pipeline(args, device)
    model_load_elapsed = time.perf_counter() - model_load_t0
    logger.info("Pipeline loaded in %.3fs", model_load_elapsed)

    # --- Warmup ---
    for wi in range(args.warmup):
        t = run_once(pipeline, args, device, width, height, run_idx=wi, save=False)
        logger.info("[Warmup %d/%d] %.3fs", wi + 1, args.warmup, t)
    if args.warmup:
        logger.info("Warmup complete.\n")

    # --- Timed runs ---
    times = []
    save = not args.no_save
    for ri in range(args.num_runs):
        t = run_once(pipeline, args, device, width, height, run_idx=ri, save=save)
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


if __name__ == "__main__":
    main()
