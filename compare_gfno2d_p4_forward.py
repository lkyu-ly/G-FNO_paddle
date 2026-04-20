from __future__ import annotations

from pathlib import Path

import numpy as np
import paddle
import torch

from generate_gfno2d_p4_random_weights import (
    DEFAULT_SEED,
    MODEL_CONFIG,
    PADDLE_WEIGHT_PATH,
    TORCH_WEIGHT_PATH,
    build_paddle_model,
    build_torch_model,
)

INPUT_SHAPE = (2, 64, 64, 10, 1)


def ensure_weight_files_exist():
    missing = [path for path in (TORCH_WEIGHT_PATH, PADDLE_WEIGHT_PATH) if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing weight files: {missing}. Run python generate_gfno2d_p4_random_weights.py first."
        )


def make_input(seed: int = DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(INPUT_SHAPE, dtype=np.float32)


def forward_torch(np_input):
    model = build_torch_model()
    state_dict = torch.load(TORCH_WEIGHT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(np_input))
    return output.detach().cpu().numpy()


def forward_paddle(np_input):
    paddle.device.set_device("cpu")
    model = build_paddle_model()
    state_dict = paddle.load(str(PADDLE_WEIGHT_PATH))
    model.set_state_dict(state_dict)
    model.eval()
    with paddle.no_grad():
        output = model(paddle.to_tensor(np_input))
    return output.numpy()


def compute_error_metrics(torch_out, paddle_out, eps: float = 1e-12):
    torch_out = np.asarray(torch_out, dtype=np.float64)
    paddle_out = np.asarray(paddle_out, dtype=np.float64)
    abs_diff = np.abs(torch_out - paddle_out)
    rel_diff = abs_diff / np.maximum(np.abs(torch_out), eps)
    return {
        "mean_abs_error": abs_diff.mean(),
        "max_abs_error": abs_diff.max(),
        "mean_rel_error": rel_diff.mean(),
        "max_rel_error": rel_diff.max(),
    }


def main():
    ensure_weight_files_exist()
    np_input = make_input()
    torch_out = forward_torch(np_input)
    paddle_out = forward_paddle(np_input)
    metrics = compute_error_metrics(torch_out, paddle_out)

    print(f"model_config={MODEL_CONFIG}")
    print(f"input_shape={np_input.shape}")
    print(f"torch_output_shape={torch_out.shape}")
    print(f"paddle_output_shape={paddle_out.shape}")
    print(f"torch_mean={torch_out.mean():.12e}")
    print(f"torch_std={torch_out.std():.12e}")
    print(f"paddle_mean={paddle_out.mean():.12e}")
    print(f"paddle_std={paddle_out.std():.12e}")
    for key, value in metrics.items():
        print(f"{key}={value:.12e}")


if __name__ == "__main__":
    main()
