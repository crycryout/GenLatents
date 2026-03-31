# GenLatents

Collected code for generating and saving pre-VAE latents from multiple text-to-video model families.

This repository packages together the local scripts and upstream patch files used to generate latent tensors and benchmark outputs for these model families:
- Wan 2.2
- CogVideoX 1.5
- HunyuanVideo 1.5
- LTX-2 / LTX-Video
- Open-Sora 2.0

## Repository layout

- `scripts/run_diffusers_video_bench.py`
  - Unified latent-export runner for `CogVideoX`, `HunyuanVideo`, and `LTX` family pipelines through `diffusers`.
- `scripts/decode_hunyuan_latents_cpu.py`
  - Standalone decoder for Hunyuan latent `.pt` files back to MP4 and stats JSON.
- `scripts/run_wan22_vbench_batch.py`
  - Batch runner for Wan 2.2 generation over a manifest, including latent export and MP4 export.
- `scripts/build_vbench_wan22_manifest.py`
  - Manifest builder for VBench prompt sampling.
- `patches/wan2.2/wan22_latent_export.patch`
  - Patch against local `Wan2.2` checkout to save final denoised latents before VAE decode and support local environment compatibility fixes.
- `patches/open-sora/opensora_latent_export.patch`
  - Patch against local `Open-Sora` checkout to save sampled latents and relax a few inference-time environment constraints.
- `examples/results/`
  - Example stats and aggregate summary JSON files generated from local benchmark runs.

## Model coverage

### 1. Diffusers-based families

The unified `scripts/run_diffusers_video_bench.py` covers:
- `CogVideoX1.5-5B`
- `HunyuanVideo-1.5-Diffusers-720p_t2v`
- `LTX-2`
- `LTX-Video`

It supports:
- exporting final latent tensors to `.pt`
- decoding frames to MP4
- writing per-run stats JSON with raw frame size, latent size, and MP4 size

### 2. Wan 2.2

Wan requires upstream source patches for latent export. The included patch adds:
- `--save_latents_file` support in `generate.py`
- latent return paths in `wan/text2video.py`
- latent return paths in `wan/textimage2video.py`
- an SDPA fallback in Wan attention when flash-attn is unavailable
- safer optional imports in `wan/__init__.py`
- single-process NCCL init for local TI2V loading compatibility

### 3. Open-Sora 2.0

Open-Sora also required upstream source patches. The included patch adds:
- latent save hook in `opensora/utils/sampling.py`
- save path wiring in `scripts/diffusion/inference.py`
- inference-friendly fallback for missing TensorNVMe async writer
- flash attention fallback to torch SDPA
- deferred policy imports to reduce unnecessary inference-time failures

## Example outputs

The `examples/results/` directory includes local benchmark result JSON files, including:
- `video_model_benchmark_summary.json`
- per-model stats for CogVideoX, HunyuanVideo, LTX-2, and Open-Sora

These are example artifacts, not required inputs.

## Usage notes

This repository does not vendor full upstream model repositories or model weights.

Expected local environment:
- upstream model repos checked out separately when required
- model weights available locally or through Hugging Face
- `ffmpeg` installed
- Python dependencies from `requirements.txt`

For Wan and Open-Sora, apply the included patch files inside the respective upstream repositories before running the generation scripts.

## Scope

This repository focuses on latent generation and latent export code paths. It does not include the separate latent compression work.
