#!/usr/bin/env python3
"""Detect GPU memory and provide conservative runtime defaults for run_main.py."""
from __future__ import annotations


def detect_card_profile(torch_module=None) -> dict:
    """Return GPU details and a context limit suited to available VRAM."""
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            torch_module = None

    profile = {
        "name": "CPU",
        "total_vram_gb": 0.0,
        "free_vram_gb": 0.0,
        "compute_capability": "n/a",
        "dtype": "float32",
        "context_multiplier": 1,
        "selection": "CPU fallback",
    }
    if torch_module is None:
        return profile

    try:
        if not torch_module.cuda.is_available():
            return profile
        device_index = torch_module.cuda.current_device()
        props = torch_module.cuda.get_device_properties(device_index)
        total_bytes = int(props.total_memory)
        try:
            free_bytes, _ = torch_module.cuda.mem_get_info(device_index)
        except Exception:
            free_bytes = total_bytes
        total_gb = total_bytes / (1024 ** 3)
        free_gb = max(0.0, min(float(free_bytes) / (1024 ** 3), total_gb))
        capability = torch_module.cuda.get_device_capability(device_index)
        try:
            native_bf16 = bool(torch_module.cuda.is_bf16_supported(including_emulation=False))
        except (TypeError, AttributeError):
            native_bf16 = capability[0] >= 8
        dtype = "bfloat16" if native_bf16 else "float32"
        multiplier = 2 if total_gb >= 12.0 and free_gb >= 8.0 else 1
        profile = {
            "name": str(props.name),
            "total_vram_gb": round(total_gb, 2),
            "free_vram_gb": round(free_gb, 2),
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "dtype": dtype,
            "context_multiplier": multiplier,
            "selection": (
                "native BF16; full context" if native_bf16 and multiplier == 2 else
                "native BF16; conservative context" if native_bf16 else
                "FP32 fallback; conservative context"
            ),
        }
    except Exception as exc:
        profile["selection"] = f"GPU detection unavailable: {type(exc).__name__}"
    return profile


def choose_asr_device(torch_module, requested: str, tts_device: str) -> str:
    """Use CUDA for ASR when auto mode has enough VRAM left after TTS loads."""
    if requested == "cpu":
        return "cpu"
    if tts_device != "cuda" or torch_module is None:
        return "cpu"
    if requested == "cuda":
        return "cuda"
    try:
        free_bytes, _ = torch_module.cuda.mem_get_info()
        return "cuda" if free_bytes >= int(2.0 * (1024 ** 3)) else "cpu"
    except Exception:
        return "cuda"


def main():
    profile = detect_card_profile()
    print(
        f"GPU: {profile['name']}; VRAM {profile['free_vram_gb']:.1f}/"
        f"{profile['total_vram_gb']:.1f} GB free/total; "
        f"compute capability {profile['compute_capability']}; "
        f"dtype={profile['dtype']}; context multiplier={profile['context_multiplier']}"
    )


if __name__ == "__main__":
    main()
