"""
Qwen Image text-to-image inference using SGLang Diffusion.

Supports two model variants:
  - Qwen/Qwen-Image         (base)
  - Qwen/Qwen-Image-2512    (latest, recommended)

Usage examples:
  # Basic text-to-image
  python run_qwen_image_t2i.py \
      --prompt "一只穿着宇航服的猫站在月球表面，背景是蓝色地球"

  # Custom resolution & parameters
  python run_qwen_image_t2i.py \
      --prompt "A futuristic city at sunset, cyberpunk style" \
      --height 1024 --width 1024 \
      --steps 50 --cfg-scale 4.0 --seed 123

  # Generate multiple images
  python run_qwen_image_t2i.py \
      --prompt "Watercolor painting of a mountain lake" \
      --num-outputs 4

  # CPU offload for low-VRAM GPUs
  python run_qwen_image_t2i.py \
      --prompt "Oil painting of a cat" \
      --text-encoder-cpu-offload --pin-cpu-memory

  # Profile performance
  python run_qwen_image_t2i.py \
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

from sglang.multimodal_gen import DiffGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen Image text-to-image with SGLang Diffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen-Image-2512",
        help="HuggingFace model path or local directory "
        "(Qwen/Qwen-Image, Qwen/Qwen-Image-2512)",
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
        default=None,
        help="Negative prompt to suppress unwanted features",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter weights",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--output-file-name",
        type=str,
        default=None,
        help="Custom output file name (without extension)",
    )

    gen_group = parser.add_argument_group("Generation parameters")
    gen_group.add_argument("--height", type=int, default=None, help="Output image height")
    gen_group.add_argument("--width", type=int, default=None, help="Output image width")
    gen_group.add_argument("--steps", type=int, default=None, help="Number of denoising steps (default: 50)")
    gen_group.add_argument("--cfg-scale", type=float, default=None, help="Guidance scale (default: 4.0)")
    gen_group.add_argument("--seed", type=int, default=42, help="Random seed")
    gen_group.add_argument("--num-outputs", type=int, default=1, help="Number of images to generate")

    perf_group = parser.add_argument_group("Performance options")
    perf_group.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    perf_group.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="Disable all CPU offload (load everything on GPU). "
        "Use when GPU VRAM is sufficient but system RAM is limited.",
    )
    perf_group.add_argument(
        "--text-encoder-cpu-offload",
        action="store_true",
        help="Offload text encoder to CPU to save GPU VRAM",
    )
    perf_group.add_argument(
        "--pin-cpu-memory",
        action="store_true",
        help="Pin CPU memory for faster data transfer",
    )

    prof_group = parser.add_argument_group("Profiler options")
    prof_group.add_argument(
        "--profile",
        action="store_true",
        help="Enable torch.profiler to trace CPU/CUDA performance",
    )
    prof_group.add_argument(
        "--profile-dir",
        type=str,
        default="./profiler_traces",
        help="Directory to save profiler trace files",
    )
    prof_group.add_argument(
        "--profile-warmup",
        type=int,
        default=0,
        help="Number of warmup runs before the profiled run (not traced)",
    )
    prof_group.add_argument(
        "--profile-repeat",
        type=int,
        default=1,
        help="Number of profiled runs (each produces a separate trace)",
    )
    prof_group.add_argument(
        "--profile-with-stack",
        action="store_true",
        help="Record Python & C++ call stacks (larger trace, more detail)",
    )
    prof_group.add_argument(
        "--profile-memory",
        action="store_true",
        help="Track CUDA memory allocation/deallocation events",
    )
    prof_group.add_argument(
        "--profile-summary",
        action="store_true",
        help="Print a table summary of top operators to stdout",
    )
    prof_group.add_argument(
        "--profile-summary-top-n",
        type=int,
        default=30,
        help="Number of top operators to show in the summary table",
    )

    return parser.parse_args()


def _build_sampling_kwargs(args) -> dict:
    sampling_kwargs = dict(
        prompt=args.prompt,
        output_path=args.output_dir,
        save_output=True,
        seed=args.seed,
        num_outputs_per_prompt=args.num_outputs,
    )
    if args.negative_prompt is not None:
        sampling_kwargs["negative_prompt"] = args.negative_prompt
    if args.height is not None:
        sampling_kwargs["height"] = args.height
    if args.width is not None:
        sampling_kwargs["width"] = args.width
    if args.steps is not None:
        sampling_kwargs["num_inference_steps"] = args.steps
    if args.cfg_scale is not None:
        sampling_kwargs["guidance_scale"] = args.cfg_scale
    if args.output_file_name is not None:
        sampling_kwargs["output_file_name"] = args.output_file_name
    return sampling_kwargs


def _run_generate(generator, sampling_kwargs, run_label="Generate"):
    """Single generation call with timing, returns (result_list, elapsed_seconds)."""
    logger.info("[%s] Starting image generation...", run_label)
    t0 = time.perf_counter()

    result = generator.generate(sampling_params_kwargs=dict(sampling_kwargs))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if result is None:
        logger.warning("[%s] Generation returned None (%.3fs elapsed)", run_label, elapsed)
        return [], elapsed
    results = result if isinstance(result, list) else [result]
    logger.info(
        "[%s] Generation complete: %d image(s) in %.3fs (%.1f img/s)",
        run_label,
        len(results),
        elapsed,
        len(results) / elapsed if elapsed > 0 else 0,
    )
    for i, r in enumerate(results):
        engine_time = r.generation_time or 0
        logger.info(
            "[%s]   #%d  engine_time=%.3fs  wall_time=%.3fs  saved=%s",
            run_label,
            i + 1,
            engine_time,
            elapsed,
            r.output_file_path,
        )
    return results, elapsed


def _print_profiler_summary(prof, args, trace_path):
    """Print operator-level summary tables and save metrics JSON."""
    separator = "=" * 80

    print(f"\n{separator}")
    print("PROFILER SUMMARY — Top CUDA operators by total CUDA time")
    print(separator)
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=args.profile_summary_top_n,
        )
    )

    print(f"\n{separator}")
    print("PROFILER SUMMARY — Top CPU operators by total CPU time")
    print(separator)
    print(
        prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=args.profile_summary_top_n,
        )
    )

    if args.profile_memory:
        print(f"\n{separator}")
        print("PROFILER SUMMARY — Top operators by CUDA memory usage")
        print(separator)
        print(
            prof.key_averages().table(
                sort_by="self_cuda_memory_usage",
                row_limit=args.profile_summary_top_n,
            )
        )

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

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Model:   %s", args.model_path)
    logger.info("Prompt:  %s", args.prompt)
    if args.negative_prompt:
        logger.info("Neg:     %s", args.negative_prompt)
    if args.profile:
        logger.info("Profiler:  ON  (warmup=%d, repeat=%d)", args.profile_warmup, args.profile_repeat)
        logger.info("  trace dir:   %s", args.profile_dir)
        logger.info("  with_stack:  %s", args.profile_with_stack)
        logger.info("  memory:      %s", args.profile_memory)
    logger.info("=" * 60)

    server_kwargs = dict(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
    )
    if args.no_cpu_offload:
        server_kwargs["dit_cpu_offload"] = False
        server_kwargs["text_encoder_cpu_offload"] = False
        server_kwargs["vae_cpu_offload"] = False
        server_kwargs["pin_cpu_memory"] = False
        logger.info("CPU offload disabled — all weights will be loaded on GPU")
    else:
        if args.text_encoder_cpu_offload:
            server_kwargs["text_encoder_cpu_offload"] = True
        if args.pin_cpu_memory:
            server_kwargs["pin_cpu_memory"] = True
    if args.lora_path:
        server_kwargs["lora_path"] = args.lora_path

    logger.info("Loading model...")
    model_load_t0 = time.perf_counter()
    generator = DiffGenerator.from_pretrained(**server_kwargs)
    model_load_elapsed = time.perf_counter() - model_load_t0
    logger.info("Model loaded in %.3fs", model_load_elapsed)

    try:
        sampling_kwargs = _build_sampling_kwargs(args)

        if not args.profile:
            results, elapsed = _run_generate(generator, sampling_kwargs, run_label="Run")
            if not results:
                logger.error("Generation failed, no output produced.")
                sys.exit(1)
            logger.info("-" * 60)
            logger.info("Summary: model_load=%.3fs  generate=%.3fs  total=%.3fs",
                        model_load_elapsed, elapsed, model_load_elapsed + elapsed)
            return

        # --- Profiling mode ---
        trace_dir = Path(args.profile_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)

        for wi in range(args.profile_warmup):
            _run_generate(generator, sampling_kwargs, run_label=f"Warmup {wi + 1}/{args.profile_warmup}")
        if args.profile_warmup:
            logger.info("Warmup complete.")

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        all_wall_times = []
        for ri in range(args.profile_repeat):
            trace_path = trace_dir / f"trace_run{ri}.json"
            logger.info("[Profile run %d/%d] tracing...", ri + 1, args.profile_repeat)

            with profile(
                activities=activities,
                record_shapes=True,
                profile_memory=args.profile_memory,
                with_stack=args.profile_with_stack,
                with_flops=True,
            ) as prof:
                with record_function("qwen_image_t2i_generate"):
                    results, wall_time = _run_generate(
                        generator, sampling_kwargs,
                        run_label=f"Profile {ri + 1}/{args.profile_repeat}",
                    )

            all_wall_times.append(wall_time)
            prof.export_chrome_trace(str(trace_path))
            logger.info("  Chrome trace saved to: %s", trace_path)
            logger.info("  -> Open in: chrome://tracing  or  https://ui.perfetto.dev")

            if args.profile_summary:
                _print_profiler_summary(prof, args, trace_path)

        if len(all_wall_times) > 1:
            avg_t = sum(all_wall_times) / len(all_wall_times)
            min_t = min(all_wall_times)
            max_t = max(all_wall_times)
            logger.info("-" * 60)
            logger.info(
                "Profile timing: avg=%.3fs  min=%.3fs  max=%.3fs  over %d runs",
                avg_t, min_t, max_t, len(all_wall_times),
            )

        if not args.profile_summary:
            logger.info("Tip: add --profile-summary to print operator-level tables to stdout.")

    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
