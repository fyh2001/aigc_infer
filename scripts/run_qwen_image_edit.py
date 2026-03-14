"""
Qwen Image Edit inference using SGLang Diffusion.

Supports three model variants:
  - Qwen/Qwen-Image-Edit       (base)
  - Qwen/Qwen-Image-Edit-2509  (plus)
  - Qwen/Qwen-Image-Edit-2511  (latest)

Usage examples:
  # Basic usage
  python run_qwen_image_edit.py \
      --model-path Qwen/Qwen-Image-Edit-2511 \
      --image-path input.jpg \
      --prompt "Turn it into a watercolor painting."

  # With LoRA adapter
  python run_qwen_image_edit.py \
      --model-path Qwen/Qwen-Image-Edit-2511 \
      --image-path input.jpg \
      --prompt "Transform into anime." \
      --lora-path prithivMLmods/Qwen-Image-Edit-2511-Anime

  # Custom output & resolution
  python run_qwen_image_edit.py \
      --model-path Qwen/Qwen-Image-Edit-2511 \
      --image-path input.jpg \
      --prompt "Add snow to the scene." \
      --output-dir results/ \
      --height 1024 --width 1024 \
      --steps 40 --cfg-scale 4.0 --seed 123

  # CPU offload for low-VRAM GPUs
  python run_qwen_image_edit.py \
      --model-path Qwen/Qwen-Image-Edit-2511 \
      --image-path input.jpg \
      --prompt "Make the sky sunset colors." \
      --text-encoder-cpu-offload

  # Profile performance (SGLang built-in profiler)
  python run_qwen_image_edit.py \
      --model-path Qwen/Qwen-Image-Edit-2511 \
      --image-path input.jpg \
      --prompt "Add snow to the scene." \
      --profile --profile-all-stages
"""

import argparse
import logging
import sys
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen Image Edit with SGLang Diffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen-Image-Edit-2511",
        help="HuggingFace model path or local directory "
        "(Qwen/Qwen-Image-Edit, Qwen/Qwen-Image-Edit-2509, Qwen/Qwen-Image-Edit-2511)",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path or URL to the input image to edit",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Editing instruction, e.g. 'Turn it into a watercolor painting.'",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt for generation",
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
    gen_group.add_argument("--steps", type=int, default=None, help="Number of denoising steps")
    gen_group.add_argument("--cfg-scale", type=float, default=None, help="Guidance scale")
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
    perf_group.add_argument(
        "--enable-torch-compile",
        action="store_true",
        help="Enable torch.compile on the DiT model for faster inference",
    )
    perf_group.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        help="Attention backend to use (e.g. flash_attn, sage_attn)",
    )
    perf_group.add_argument(
        "--warmup-engine",
        action="store_true",
        help="Run a warmup pass through the engine before generation",
    )

    prof_group = parser.add_argument_group("Profiler options")
    prof_group.add_argument(
        "--profile",
        action="store_true",
        help="Enable SGLang built-in profiler (torch.profiler inside worker)",
    )
    prof_group.add_argument(
        "--profile-all-stages",
        action="store_true",
        help="Profile all stages (text encode, denoise, VAE decode)",
    )
    prof_group.add_argument(
        "--num-profiled-timesteps",
        type=int,
        default=None,
        help="Number of denoising timesteps to profile (default: all)",
    )
    prof_group.add_argument(
        "--profile-warmup",
        type=int,
        default=0,
        help="Number of warmup runs (without profiling) before the profiled run",
    )

    return parser.parse_args()


def _build_sampling_kwargs(args) -> dict:
    sampling_kwargs = dict(
        prompt=args.prompt,
        image_path=args.image_path,
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
    if args.profile:
        sampling_kwargs["profile"] = True
    if args.profile_all_stages:
        sampling_kwargs["profile_all_stages"] = True
    if args.num_profiled_timesteps is not None:
        sampling_kwargs["num_profiled_timesteps"] = args.num_profiled_timesteps
    return sampling_kwargs


def _run_generate(generator, sampling_kwargs, run_label="Generate"):
    """Single generation call with timing, returns the result list."""
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
        if isinstance(r, dict):
            engine_time = r.get("generation_time", 0) or 0
            output_path = r.get("output_file_path", "N/A")
        else:
            engine_time = r.generation_time or 0
            output_path = r.output_file_path
        logger.info(
            "[%s]   #%d  engine_time=%.3fs  wall_time=%.3fs  saved=%s",
            run_label,
            i + 1,
            engine_time,
            elapsed,
            output_path,
        )
    return results, elapsed


def main():
    args = parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Model:   %s", args.model_path)
    logger.info("Image:   %s", args.image_path)
    logger.info("Prompt:  %s", args.prompt)
    if args.enable_torch_compile:
        logger.info("torch.compile:  ON")
    if args.attention_backend:
        logger.info("Attention:  %s", args.attention_backend)
    if args.profile:
        logger.info("Profiler:  ON  (SGLang built-in, profile_all_stages=%s)", args.profile_all_stages)
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
    if args.enable_torch_compile:
        server_kwargs["enable_torch_compile"] = True
    if args.attention_backend:
        server_kwargs["attention_backend"] = args.attention_backend
    if args.warmup_engine:
        server_kwargs["warmup_engine"] = True

    logger.info("Loading model...")
    model_load_t0 = time.perf_counter()
    generator = DiffGenerator.from_pretrained(**server_kwargs)
    model_load_elapsed = time.perf_counter() - model_load_t0
    logger.info("Model loaded in %.3fs", model_load_elapsed)

    try:
        sampling_kwargs = _build_sampling_kwargs(args)

        if args.profile and args.profile_warmup > 0:
            warmup_kwargs = {k: v for k, v in sampling_kwargs.items()
                            if k not in ("profile", "profile_all_stages", "num_profiled_timesteps")}
            for wi in range(args.profile_warmup):
                _run_generate(generator, warmup_kwargs,
                              run_label=f"Warmup {wi + 1}/{args.profile_warmup}")
            logger.info("Warmup complete, starting profiled run.")

        results, elapsed = _run_generate(generator, sampling_kwargs, run_label="Run")
        if not results:
            logger.error("Generation failed, no output produced.")
            sys.exit(1)
        logger.info("-" * 60)
        logger.info("Summary: model_load=%.3fs  generate=%.3fs  total=%.3fs",
                    model_load_elapsed, elapsed, model_load_elapsed + elapsed)

    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
