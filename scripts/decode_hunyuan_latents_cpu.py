import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLHunyuanVideo15
from diffusers.utils import export_to_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--latent-file", required=True)
    parser.add_argument("--mp4-file", required=True)
    parser.add_argument("--stats-file", required=True)
    args = parser.parse_args()

    latent_path = Path(args.latent_file)
    mp4_path = Path(args.mp4_file)
    stats_path = Path(args.stats_file)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(latent_path, map_location="cpu")
    latents = payload["latents"].to(torch.float32)
    fps = int(payload["fps"])

    vae = AutoencoderKLHunyuanVideo15.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    with torch.inference_mode():
        decoded = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]

    decoded = decoded.detach().cpu().clamp(-1, 1)
    decoded = ((decoded + 1.0) / 2.0 * 255.0).to(torch.uint8)
    frames = decoded[0].permute(1, 2, 3, 0).numpy()
    frame_list = [np.asarray(frame) for frame in frames]
    export_to_video(frame_list, str(mp4_path), fps=fps)

    first = frame_list[0]
    raw_video_bytes = len(frame_list) * first.size * first.itemsize
    stats = {
        "family": payload["family"],
        "model_id": payload["model_id"],
        "prompt": payload["prompt"],
        "seed": payload["seed"],
        "width": payload["width"],
        "height": payload["height"],
        "num_frames": len(frame_list),
        "fps": fps,
        "frame_shape": list(first.shape),
        "frame_dtype": str(first.dtype),
        "raw_video_bytes": raw_video_bytes,
        "raw_video_mib": raw_video_bytes / 1024 / 1024,
        "mp4_bytes": mp4_path.stat().st_size,
        "mp4_mib": mp4_path.stat().st_size / 1024 / 1024,
        "latents_shape": list(latents.shape),
        "latents_dtype": str(latents.dtype),
        "latents_raw_bytes": latents.numel() * latents.element_size(),
        "latents_raw_mib": (latents.numel() * latents.element_size()) / 1024 / 1024,
        "latents_pt_bytes": latent_path.stat().st_size,
        "latents_pt_mib": latent_path.stat().st_size / 1024 / 1024,
        "mp4_path": str(mp4_path),
        "latents_path": str(latent_path),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
