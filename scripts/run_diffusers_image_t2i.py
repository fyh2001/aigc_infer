"""
Qwen Image text-to-image inference using native HuggingFace Diffusers.
For benchmarking comparison with the SGLang version (run_qwen_image_t2i.py).

Supports:
  - Qwen/Qwen-Image-2512  (latest, recommended)
  - Qwen/Qwen-Image       (base)

Usage examples:
  # Basic text-to-image
  python run_diffusers_image_t2i.py \
      --prompt "A cat astronaut on the moon"

  # Custom resolution (16:9)
  python run_diffusers_image_t2i.py \
      --prompt "A futuristic city at sunset" \
      --aspect-ratio 16:9 --steps 50 --seed 123

  # Explicit height/width (overrides aspect-ratio)
  python run_diffusers_image_t2i.py \
      --prompt "Oil painting of a lake" \
      --height 1024 --width 1024

  # Profile performance
  python run_diffusers_image_t2i.py \
      --prompt "A beautiful landscape" \
      --profile --profile-summary
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

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
        description="Run Qwen Image T2I with native Diffusers (baseline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen-Image-2512",
        help="HuggingFace model path or local directory (Qwen/Qwen-Image, Qwen/Qwen-Image-2512)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the image to generate",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
        help="Negative prompt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_diffusers/",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--output-file-name",
        type=str,
        default=None,
        help="Custom output file name (without extension)",
    )

    gen_group = parser.add_argument_group("Generation parameters")
    gen_group.add_argument(
        "--aspect-ratio",
        type=str,
        default="1:1",
        choices=list(ASPECT_RATIOS.keys()),
        help="Aspect ratio preset (ignored if --height/--width are set)",
    )
    gen_group.add_argument("--height", type=int, default=None, help="Output image height (overrides --aspect-ratio)")
    gen_group.add_argument("--width", type=int, default=None, help="Output image width (overrides --aspect-ratio)")
    gen_group.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    gen_group.add_argument("--cfg-scale", type=float, default=4.0, help="true_cfg_scale for prompt adherence")
    gen_group.add_argument("--seed", type=int, default=42, help="Random seed")
    gen_group.add_argument("--num-outputs", type=int, default=1, help="Number of images to generate")

    perf_group = parser.add_argument_group("Performance options")
    perf_group.add_argument("--cpu-offload", action="store_true", help="Enable model CPU offload")
    perf_group.add_argument("--device", type=str, default=None, help="Device (auto-detected if not set)")

    prof_group = parser.add_argument_group("Profiler options")
    prof_group.add_argument("--profile", action="store_true", help="Enable torch.profiler")
    prof_group.add_argument("--profile-dir", type=str, default="./profiler_traces", help="Trace output directory")
    prof_group.add_argument("--profile-warmup", type=int, default=0, help="Warmup runs before profiling")
    prof_group.add_argument("--profile-repeat", type=int, default=1, help="Number of profiled runs")
    prof_group.add_argument("--profile-with-stack", action="store_true", help="Record call stacks")
    prof_group.add_argument("--profile-memory", action="store_true", help="Track CUDA memory events")
    prof_group.add_argument("--profile-summary", action="store_true", help="Print operator summary tables")
    prof_group.add_argument("--profile-summary-top-n", type=int, default=30, help="Top N operators in summary")

    return parser.parse_args()


def _resolve_device(args):
    if args.device:
        return args.device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dimensions(args):
    if args.height is not None and args.width is not None:
        return args.width, args.height
    w, h = ASPECT_RATIOS[args.aspect_ratio]
    return w, h


def _load_pipeline(args, device):
    from diffusers import DiffusionPipeline

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    logger.info("Pipeline class: DiffusionPipeline (auto)")
    logger.info("dtype: %s, device: %s", dtype, device)

    pipeline = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=dtype)

    if args.cpu_offload and device == "cuda":
        pipeline.enable_model_cpu_offload()
        logger.info("Model CPU offload enabled")
    else:
        pipeline.to(device)

    return pipeline


def _run_generate(pipeline, args, device, width, height, run_label="Generate"):
    logger.info("[%s] Starting image generation (%dx%d)...", run_label, width, height)

    gen_device = device if device == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(args.seed)

    t0 = time.perf_counter()
    with torch.inference_mode():
        output = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=args.steps,
            true_cfg_scale=args.cfg_scale,
            num_images_per_prompt=args.num_outputs,
            generator=generator,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    result_images = output.images
    logger.info(
        "[%s] Generation complete: %d image(s) in %.3fs (%.1f img/s)",
        run_label,
        len(result_images),
        elapsed,
        len(result_images) / elapsed if elapsed > 0 else 0,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for i, img in enumerate(result_images):
        if args.output_file_name:
            fname = f"{args.output_file_name}_{i}.png" if len(result_images) > 1 else f"{args.output_file_name}.png"
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            fname = f"t2i_{timestamp}_s{args.seed}_{i}.png"
        out_path = output_dir / fname
        img.save(str(out_path))
        saved_paths.append(out_path)
        logger.info("[%s]   #%d  wall_time=%.3fs  saved=%s", run_label, i + 1, elapsed, out_path)

    return result_images, elapsed, saved_paths


def _print_profiler_summary(prof, args, trace_path):
    separator = "=" * 80

    print(f"\n{separator}")
    print("PROFILER SUMMARY — Top CUDA operators by total CUDA time")
    print(separator)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.profile_summary_top_n))

    print(f"\n{separator}")
    print("PROFILER SUMMARY — Top CPU operators by total CPU time")
    print(separator)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=args.profile_summary_top_n))

    if args.profile_memory:
        print(f"\n{separator}")
        print("PROFILER SUMMARY — Top operators by CUDA memory usage")
        print(separator)
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=args.profile_summary_top_n))

    key_metrics = []
    for evt in prof.key_averages():
        key_metrics.append({
            "name": evt.key,
            "count": evt.count,
            "cpu_time_total_us": evt.cpu_time_total,
            "cuda_time_total_us": evt.cuda_time_total,
            "cpu_time_avg_us": evt.cpu_time_total / max(evt.count, 1),
            "cuda_time_avg_us": evt.cuda_time_total / max(evt.count, 1),
            "self_cpu_time_us": evt.self_cpu_time_total,
            "self_cuda_time_us": evt.self_cuda_time_total,
        })
    key_metrics.sort(key=lambda x: x["cuda_time_total_us"], reverse=True)

    metrics_path = trace_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(key_metrics[:args.profile_summary_top_n * 2], f, indent=2)
    print(f"\nMetrics JSON saved to: {metrics_path}")


def main():
    args = parse_args()
    device = _resolve_device(args)
    width, height = _resolve_dimensions(args)

    logger.info("=" * 60)
    logger.info("Backend:  Diffusers (baseline)")
    logger.info("Model:    %s", args.model_path)
    logger.info("Prompt:   %s", args.prompt)
    logger.info("Size:     %dx%d", width, height)
    logger.info("Device:   %s", device)
    if args.profile:
        logger.info("Profiler: ON  (warmup=%d, repeat=%d)", args.profile_warmup, args.profile_repeat)
    logger.info("=" * 60)

    logger.info("Loading pipeline...")
    model_load_t0 = time.perf_counter()
    pipeline = _load_pipeline(args, device)
    model_load_elapsed = time.perf_counter() - model_load_t0
    logger.info("Pipeline loaded in %.3fs", model_load_elapsed)

    if not args.profile:
        _, elapsed, _ = _run_generate(pipeline, args, device, width, height, run_label="Run")
        logger.info("-" * 60)
        logger.info("Summary: model_load=%.3fs  generate=%.3fs  total=%.3fs",
                    model_load_elapsed, elapsed, model_load_elapsed + elapsed)
        return

    # --- Profiling mode ---
    trace_dir = Path(args.profile_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    for wi in range(args.profile_warmup):
        _run_generate(pipeline, args, device, width, height,
                     run_label=f"Warmup {wi + 1}/{args.profile_warmup}")
    if args.profile_warmup:
        logger.info("Warmup complete.")

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    all_wall_times = []
    for ri in range(args.profile_repeat):
        trace_path = trace_dir / f"diffusers_t2i_trace_run{ri}.json"
        logger.info("[Profile run %d/%d] tracing...", ri + 1, args.profile_repeat)

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=args.profile_memory,
            with_stack=args.profile_with_stack,
            with_flops=True,
        ) as prof:
            with record_function("diffusers_t2i_generate"):
                _, wall_time, _ = _run_generate(
                    pipeline, args, device, width, height,
                    run_label=f"Profile {ri + 1}/{args.profile_repeat}",
                )

        all_wall_times.append(wall_time)
        prof.export_chrome_trace(str(trace_path))
        logger.info("  Chrome trace saved to: %s", trace_path)

        if args.profile_summary:
            _print_profiler_summary(prof, args, trace_path)

    if len(all_wall_times) > 1:
        avg_t = sum(all_wall_times) / len(all_wall_times)
        logger.info("-" * 60)
        logger.info(
            "Profile timing: avg=%.3fs  min=%.3fs  max=%.3fs  over %d runs",
            avg_t, min(all_wall_times), max(all_wall_times), len(all_wall_times),
        )

    if not args.profile_summary:
        logger.info("Tip: add --profile-summary to print operator-level tables.")


if __name__ == "__main__":
    main()
