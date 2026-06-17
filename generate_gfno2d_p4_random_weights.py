from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import os
import sys

import numpy as np
import paddle
import torch

PROJ_ROOT = Path(__file__).resolve().parent
TORCH_ROOT = PROJ_ROOT / "G-FNO"
PADDLE_ROOT = PROJ_ROOT / "G-FNO_paddle"
MODEL_DIR = PROJ_ROOT / "models"
TORCH_WEIGHT_PATH = MODEL_DIR / "gfno2d_p4_random.pt"
PADDLE_WEIGHT_PATH = MODEL_DIR / "gfno2d_p4_random.pdparams"

MODEL_CONFIG = {
    "num_channels": 1,
    "initial_step": 10,
    "modes": 12,
    "width": 10,
    "reflection": False,
    "grid_type": "symmetric",
}
DEFAULT_SEED = 20260420
ALLOWED_PADDLE_ONLY_KEYS = set()


def _reset_project_modules() -> None:
    for name in list(sys.modules):
        if name == "models" or name.startswith("models.") or name == "utils" or name.startswith("utils."):
            sys.modules.pop(name, None)


def build_torch_model():
    _reset_project_modules()
    sys.path.insert(0, os.fspath(TORCH_ROOT))
    try:
        from models.GFNO import GFNO2d as TorchGFNO2d
    finally:
        sys.path.pop(0)
    model = TorchGFNO2d(**MODEL_CONFIG)
    model.eval()
    return model


def build_paddle_model():
    _reset_project_modules()
    sys.path.insert(0, os.fspath(PADDLE_ROOT))
    try:
        from models.GFNO import GFNO2d as PaddleGFNO2d
    finally:
        sys.path.pop(0)
    paddle.device.set_device("cpu")
    model = PaddleGFNO2d(**MODEL_CONFIG)
    model.eval()
    return model


# Keys whose torch values are complex64 (N-dim) and paddle values are float32 (N+1-dim, last dim=2).
# 00_modes is NOT included — it is float32 on both sides and needs no special handling.
COMPLEX_PARAM_KEYS = {
    "conv0.conv.W.y0_modes",
    "conv0.conv.W.yposx_modes",
    "conv1.conv.W.y0_modes",
    "conv1.conv.W.yposx_modes",
    "conv2.conv.W.y0_modes",
    "conv2.conv.W.yposx_modes",
    "conv3.conv.W.y0_modes",
    "conv3.conv.W.yposx_modes",
}


def to_numpy_state_dict(state_dict):
    result = OrderedDict()
    for key, value in state_dict.items():
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        result[key] = np.asarray(value)
    return result


def diff_state_keys(torch_state, paddle_state):
    torch_keys = set(torch_state.keys())
    paddle_keys = set(paddle_state.keys())
    shared = sorted(torch_keys & paddle_keys)
    torch_only = sorted(torch_keys - paddle_keys)
    paddle_only = sorted(paddle_keys - torch_keys)
    return shared, torch_only, paddle_only


def validate_state_dict_layout(torch_state, paddle_state):
    shared_keys, torch_only_keys, paddle_only_keys = diff_state_keys(torch_state, paddle_state)
    if torch_only_keys:
        raise ValueError(f"Unexpected torch-only keys: {torch_only_keys}")

    unexpected_paddle_only = sorted(set(paddle_only_keys) - ALLOWED_PADDLE_ONLY_KEYS)
    if unexpected_paddle_only:
        raise ValueError(f"Unexpected paddle-only keys: {unexpected_paddle_only}")

    for key in shared_keys:
        t_shape = tuple(torch_state[key].shape)
        p_shape = tuple(paddle_state[key].shape)
        if key in COMPLEX_PARAM_KEYS:
            # Torch complex64 (N-dim) vs Paddle float32 (N+1-dim, last dim=2)
            if p_shape != t_shape + (2,):
                raise ValueError(
                    f"Shape mismatch for complex param {key}: "
                    f"torch={t_shape} paddle={p_shape} (expected paddle={t_shape + (2,)})"
                )
        else:
            if t_shape != p_shape:
                raise ValueError(
                    f"Shape mismatch for {key}: torch={t_shape} paddle={p_shape}"
                )

    return {
        "shared_keys": shared_keys,
        "torch_only_keys": torch_only_keys,
        "paddle_only_keys": paddle_only_keys,
    }


def convert_torch_state_to_paddle(torch_state, paddle_state):
    summary = validate_state_dict_layout(torch_state, paddle_state)
    converted = OrderedDict()
    for key, value in paddle_state.items():
        if key in torch_state:
            torch_val = torch_state[key]
            if key in COMPLEX_PARAM_KEYS:
                # Convert complex64 -> float32 with last dim=2 (real, imag)
                arr = np.asarray(torch_val)
                real_imag = np.stack([arr.real, arr.imag], axis=-1)
                converted[key] = paddle.to_tensor(real_imag.astype(np.float32))
            else:
                converted[key] = paddle.to_tensor(torch_val)
        else:
            converted[key] = value
    return converted, summary


def save_random_weights(seed: int = DEFAULT_SEED):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    torch_model = build_torch_model()
    torch.save(torch_model.state_dict(), TORCH_WEIGHT_PATH)

    paddle_model = build_paddle_model()
    torch_state = to_numpy_state_dict(torch_model.state_dict())
    paddle_state = to_numpy_state_dict(paddle_model.state_dict())
    converted_state, summary = convert_torch_state_to_paddle(torch_state, paddle_state)
    paddle.save(converted_state, os.fspath(PADDLE_WEIGHT_PATH))

    return summary


def main():
    summary = save_random_weights()
    print(f"torch_weight_path={TORCH_WEIGHT_PATH}")
    print(f"paddle_weight_path={PADDLE_WEIGHT_PATH}")
    print(f"shared_key_count={len(summary['shared_keys'])}")
    print(f"paddle_only_keys={summary['paddle_only_keys']}")


if __name__ == "__main__":
    main()
