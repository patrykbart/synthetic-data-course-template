"""Generate Egyptian Mau images with a diffusion model and record every attempt.

    uv run synth-generate --name m2-base
    uv run synth-generate --name m2-guidance-3 --guidance 3
    uv run synth-generate --name m3-strength-0.5 --mode img2img --strength 0.5
    uv run synth-generate --name m3-ip-0.5 --mode ip-adapter --ip-adapter-scale 0.5
    uv run synth-generate --name m4-backgrounds --mode inpaint
    uv run synth-generate --name m4-outpaint-1.5 --mode outpaint --outpaint-scale 1.5

Images go to pools/<name>/images/, one manifest line per attempt to
pools/<name>/manifest.jsonl, and the first 24 attempts to contact-sheet.jpg.
Running the same command again adds only the missing images.
"""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from synthdata.config import POOLS_DIR
from synthdata.data import minority_training_photos
from synthdata.inputs import (
    MIN_PET_SHARE,
    SIZE,
    background_mask,
    keep_cat,
    outpaint_inputs,
    paste_back,
    pet_share,
    source_photo,
)
from synthdata.pools import read_manifest
from synthdata.sheets import contact_sheet

MODES = ("t2i", "img2img", "ip-adapter", "inpaint", "outpaint")
SOURCE_MODES = ("img2img", "ip-adapter", "inpaint", "outpaint")
EDIT_MODES = ("inpaint", "outpaint")
SCHEDULERS = ("default", "ddim", "dpm", "euler")

# Exact model versions, so every student generates with the same weights.
MODELS = {
    "sd15": {
        "repo": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "revision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        "variant": "fp16",
        "steps": 30,
        "guidance": 7.5,
        "scheduler": "dpm",
    },
    "sdxl-turbo": {
        "repo": "stabilityai/sdxl-turbo",
        "revision": "71153311d3dbb46851df1931d3ca6e939de83304",
        "variant": "fp16",
        "steps": 4,
        "guidance": 0.0,
        "scheduler": "default",
    },
}
INPAINTING = {
    "repo": "stable-diffusion-v1-5/stable-diffusion-inpainting",
    "revision": "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb",
    "variant": "fp16",
}
IP_ADAPTER = {
    "repo": "h94/IP-Adapter",
    "subfolder": "models",
    "weight_name": "ip-adapter_sd15.safetensors",
    "revision": "018e402774aeeddd60609b4ecdb7e298259dc729",
}

PROMPTS = [
    "a candid photograph of an Egyptian Mau cat, silver spotted coat and green "
    "eyes, natural indoor light, ordinary home background",
    "a documentary photograph of an Egyptian Mau cat, slender body and naturally "
    "spotted silver coat, outdoor daylight",
    "an unedited pet photograph of an Egyptian Mau cat sitting in a home, "
    "realistic fur, natural lighting",
    "a realistic snapshot of a silver Egyptian Mau cat standing, entire cat "
    "visible, everyday background",
]
SCENE_PROMPTS = [
    "a photograph of an Egyptian Mau cat on a sofa in a living room",
    "a photograph of an Egyptian Mau cat in a garden with green grass",
    "a photograph of an Egyptian Mau cat on a wooden kitchen floor",
    "a photograph of an Egyptian Mau cat on a bed with white sheets",
    "a photograph of an Egyptian Mau cat outdoors on a stone path",
    "a photograph of an Egyptian Mau cat on a windowsill in daylight",
]
NEGATIVE_PROMPT = (
    "illustration, painting, cartoon, text, watermark, multiple cats, "
    "deformed animal, extra limbs"
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Egyptian Mau images into pools/<name>/."
    )
    parser.add_argument("--name", required=True, help="pool name, e.g. m2-base")
    parser.add_argument("--mode", choices=MODES, default="t2i")
    parser.add_argument("--model", choices=tuple(MODELS), default="sd15")
    parser.add_argument("--count", type=int, default=110, help="images in the pool")
    parser.add_argument(
        "--prompt", action="append", help="repeat the option to cycle through prompts"
    )
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--steps", type=int, help="denoising steps of the sampler")
    parser.add_argument("--guidance", type=float, help="classifier-free guidance scale")
    parser.add_argument("--scheduler", choices=SCHEDULERS, help="sampler algorithm")
    parser.add_argument(
        "--seed", type=int, default=1000, help="seed of image 0; image i uses seed + i"
    )
    parser.add_argument(
        "--strength",
        type=float,
        help="img2img: share of the photo replaced by noise (0-1); inpaint: 1.0",
    )
    parser.add_argument(
        "--ip-adapter-scale",
        type=float,
        help="ip-adapter: how strongly the photo steers the result (0-1)",
    )
    parser.add_argument(
        "--outpaint-scale", type=float, help="outpaint: how far to zoom out (1.25-2)"
    )
    parser.add_argument("--low-vram", action="store_true", help="for GPUs under 8 GB")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument(
        "--sheet-only", action="store_true", help="only redraw the contact sheet"
    )
    args = parser.parse_args(argv)
    check(args, parser)
    return args


def check(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    mode = args.mode
    if mode in ("ip-adapter", *EDIT_MODES) and args.model != "sd15":
        parser.error(f"--mode {mode} works with --model sd15 only")
    if args.strength is not None and mode not in ("img2img", "inpaint"):
        parser.error("--strength is used only with --mode img2img or inpaint")
    if args.ip_adapter_scale is not None and mode != "ip-adapter":
        parser.error("--ip-adapter-scale is used only with --mode ip-adapter")
    if args.outpaint_scale is not None and mode != "outpaint":
        parser.error("--outpaint-scale is used only with --mode outpaint")
    if mode == "img2img" and args.strength is None:
        parser.error("--mode img2img needs --strength, e.g. 0.5")
    if mode == "inpaint" and args.strength is None:
        args.strength = 1.0
    if args.strength is not None and not 0 < args.strength <= 1:
        parser.error("--strength must be in (0, 1]")
    if mode == "ip-adapter" and args.ip_adapter_scale is None:
        parser.error("--mode ip-adapter needs --ip-adapter-scale, e.g. 0.5")
    if args.ip_adapter_scale is not None and not 0 <= args.ip_adapter_scale <= 1.5:
        parser.error("--ip-adapter-scale must be between 0 and 1.5")
    if mode == "outpaint" and args.outpaint_scale is None:
        args.outpaint_scale = 1.5
    if args.outpaint_scale is not None and not 1 < args.outpaint_scale <= 3:
        parser.error("--outpaint-scale must be above 1 and at most 3")
    if args.count < 1:
        parser.error("--count must be positive")
    steps = args.steps or MODELS[args.model]["steps"]
    if mode == "img2img" and int(steps * args.strength) < 1:
        parser.error("steps × strength must be at least 1; raise --steps")


def generation_sources(mode: str) -> tuple[list[str], list[str]]:
    """Training photos to start from, and those skipped for a broken mask."""
    if mode not in SOURCE_MODES:
        return [], []
    photos = minority_training_photos()
    if mode != "inpaint":
        return photos, []
    usable = [photo for photo in photos if pet_share(photo) >= MIN_PET_SHARE]
    return usable, [photo for photo in photos if photo not in usable]


def pool_settings(args: argparse.Namespace) -> dict:
    defaults = MODELS[args.model]
    spec = INPAINTING if args.mode in EDIT_MODES else defaults
    prompts = args.prompt or (SCENE_PROMPTS if args.mode in EDIT_MODES else PROMPTS)
    sources, skipped = generation_sources(args.mode)
    return {
        "mode": args.mode,
        "model": args.model,
        "repo": spec["repo"],
        "revision": spec["revision"],
        "variant": spec["variant"],
        "adapter": IP_ADAPTER if args.mode == "ip-adapter" else None,
        "scheduler": args.scheduler or defaults["scheduler"],
        "prompts": prompts,
        "negative_prompt": args.negative_prompt,
        "steps": args.steps or defaults["steps"],
        "guidance": defaults["guidance"] if args.guidance is None else args.guidance,
        "strength": args.strength,
        "ip_adapter_scale": args.ip_adapter_scale,
        "outpaint_scale": args.outpaint_scale,
        "seed": args.seed,
        "size": SIZE,
        "sources": sources,
        "skipped_sources": skipped,
    }


def check_pool(pool_dir: Path, settings: dict) -> None:
    """One set of settings per pool, so every image in it has the same recipe."""
    path = pool_dir / "pool.json"
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise SystemExit(
                f"{pool_dir} was made with different settings; "
                "choose a new --name for a new experiment"
            )
    else:
        pool_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, indent=2) + "\n")


def effective_steps(settings: dict) -> int:
    """img2img and inpainting skip the first part of the schedule."""
    if settings["mode"] in ("img2img", "inpaint"):
        return min(int(settings["steps"] * settings["strength"]), settings["steps"])
    return settings["steps"]


def pick_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pipeline(settings: dict, device, dtype, low_vram: bool):
    from diffusers import (
        AutoPipelineForImage2Image,
        AutoPipelineForInpainting,
        AutoPipelineForText2Image,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
    )
    from diffusers.utils import logging

    logging.set_verbosity_error()

    classes = {
        "t2i": AutoPipelineForText2Image,
        "ip-adapter": AutoPipelineForText2Image,
        "img2img": AutoPipelineForImage2Image,
        "inpaint": AutoPipelineForInpainting,
        "outpaint": AutoPipelineForInpainting,
    }
    pipe = classes[settings["mode"]].from_pretrained(
        settings["repo"],
        revision=settings["revision"],
        variant=settings["variant"],
        torch_dtype=dtype,
    )
    schedulers = {
        "ddim": DDIMScheduler,
        "dpm": DPMSolverMultistepScheduler,
        "euler": EulerDiscreteScheduler,
    }
    if settings["scheduler"] != "default":
        scheduler = schedulers[settings["scheduler"]]
        pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
    if settings["adapter"]:
        adapter = settings["adapter"]
        pipe.load_ip_adapter(
            adapter["repo"],
            subfolder=adapter["subfolder"],
            weight_name=adapter["weight_name"],
            revision=adapter["revision"],
        )
        pipe.set_ip_adapter_scale(settings["ip_adapter_scale"])
    if low_vram and device.type == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    if low_vram:
        pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_one(pipe, index: int, settings: dict, context: dict) -> dict:
    import diffusers
    import torch

    mode = settings["mode"]
    seed = settings["seed"] + index
    prompt = settings["prompts"][index % len(settings["prompts"])]
    call = {
        "prompt": prompt,
        "negative_prompt": settings["negative_prompt"] or None,
        "num_inference_steps": settings["steps"],
        "guidance_scale": settings["guidance"],
        "generator": torch.Generator("cpu").manual_seed(seed),
    }
    source = photo = None
    if settings["sources"]:
        source = settings["sources"][index % len(settings["sources"])]
        photo = context["photos"][source]
    if mode in ("t2i", "ip-adapter"):
        call.update(width=SIZE, height=SIZE)
    if mode == "ip-adapter":
        call["ip_adapter_image"] = photo
    if mode == "img2img":
        call.update(image=photo, strength=settings["strength"])
    if mode == "inpaint":
        mask = context["masks"][source]
        call.update(image=photo, mask_image=mask, strength=settings["strength"])
    if mode == "outpaint":
        canvas, mask, small, offset = outpaint_inputs(photo, settings["outpaint_scale"])
        call.update(image=canvas, mask_image=mask, strength=1.0)
    if mode in EDIT_MODES:
        call.update(width=SIZE, height=SIZE)

    device = context["device"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = pipe(**call)
    seconds = time.perf_counter() - started
    image = result.images[0]
    flags = getattr(result, "nsfw_content_detected", None)
    blocked = bool(flags and flags[0])
    if not blocked and mode == "inpaint":
        image = keep_cat(photo, image, context["masks"][source])
    if not blocked and mode == "outpaint":
        image = paste_back(image, small, offset)

    sample = f"{context['pool_dir'].name}-{index:04d}"
    path = context["pool_dir"] / "images" / f"{sample}.png"
    temporary = path.with_suffix(".tmp.png")
    image.save(temporary)
    temporary.replace(path)
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    return {
        "sample_id": sample,
        "file": f"images/{sample}.png",
        "status": "safety_blocked" if blocked else "ok",
        "mode": mode,
        "model": settings["model"],
        "model_repo": settings["repo"],
        "model_revision": settings["revision"],
        "adapter": settings["adapter"],
        "scheduler": type(pipe.scheduler).__name__,
        "prompt": prompt,
        "negative_prompt": settings["negative_prompt"],
        "seed": seed,
        "steps": settings["steps"],
        "effective_steps": effective_steps(settings),
        "guidance": settings["guidance"],
        "strength": settings["strength"],
        "ip_adapter_scale": settings["ip_adapter_scale"],
        "outpaint_scale": settings["outpaint_scale"],
        "source": source,
        "width": image.width,
        "height": image.height,
        "dtype": str(context["dtype"]).removeprefix("torch."),
        "device": device.type,
        "gpu": context["gpu"],
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "seconds": round(seconds, 2),
        "peak_vram_mb": round(peak) if peak else None,
        "sha256": sha256(path),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    settings = pool_settings(args)
    for photo in settings["skipped_sources"]:
        print(
            f"Skipping {photo}: its annotation marks no cat, so inpainting "
            "would repaint the whole photo."
        )
    pool_dir = POOLS_DIR / args.name
    check_pool(pool_dir, settings)
    (pool_dir / "images").mkdir(exist_ok=True)
    done = {
        record["sample_id"]
        for record in read_manifest(pool_dir)
        if (pool_dir / record["file"]).exists()
    }
    todo = [
        index for index in range(args.count) if f"{args.name}-{index:04d}" not in done
    ]
    if todo and not args.sheet_only:
        import torch

        device = pick_device(args.device)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        print(
            f"{args.name}: {len(todo)} of {args.count} images, {args.mode} with "
            f"{settings['repo']} on {device} (the first run downloads the model)"
        )
        pipe = load_pipeline(settings, device, dtype, args.low_vram)
        sources = settings["sources"]
        context = {
            "pool_dir": pool_dir,
            "device": device,
            "dtype": dtype,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "photos": {name: source_photo(name) for name in sources},
            "masks": {name: background_mask(name) for name in sources}
            if args.mode == "inpaint"
            else {},
        }
        with (pool_dir / "manifest.jsonl").open("a") as manifest:
            for position, index in enumerate(todo, start=1):
                record = generate_one(pipe, index, settings, context)
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                print(
                    f"  [{position}/{len(todo)}] {record['sample_id']} "
                    f"{record['status']} {record['seconds']:.1f} s"
                )

    records = read_manifest(pool_dir)
    if records:
        sheet = contact_sheet(pool_dir, records)
        usable = sum(record["status"] == "ok" for record in records)
        print(f"{args.name}: {usable} usable images of {len(records)}; see {sheet}")


if __name__ == "__main__":
    main()
