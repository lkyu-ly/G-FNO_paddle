import sys
from pathlib import Path

import paddle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PADDLE_ROOT = PROJECT_ROOT / "G-FNO_paddle"
if str(PADDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(PADDLE_ROOT))

import paddle_utils


def test_normalize_device_spec_handles_cuda_aliases():
    assert paddle_utils.normalize_device_spec("cuda") == "gpu"
    assert paddle_utils.normalize_device_spec("cuda:1") == "gpu:1"
    assert paddle_utils.normalize_device_spec("gcu:0") == "gcu:0"
    assert paddle_utils.normalize_device_spec("npu") == "npu"
    assert paddle_utils.normalize_device_spec(None) is None


def test_resolve_runtime_device_auto_prefers_custom_devices(monkeypatch):
    monkeypatch.setattr(paddle_utils.paddle.device, "get_device", lambda: "cpu")
    monkeypatch.setattr(
        paddle_utils.paddle.device, "get_all_custom_device_type", lambda: ["gcu"]
    )
    calls = []

    def fake_set_device(device):
        calls.append(device)
        if device == "gcu":
            return device
        raise ValueError(device)

    monkeypatch.setattr(paddle_utils.paddle, "set_device", fake_set_device)

    assert paddle_utils.resolve_runtime_device("auto") == "gcu"
    assert calls == ["gcu"]


def test_resolve_runtime_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(paddle_utils.paddle.device, "get_device", lambda: "cpu")
    monkeypatch.setattr(
        paddle_utils.paddle.device, "get_all_custom_device_type", lambda: []
    )
    calls = []

    def fake_set_device(device):
        calls.append(device)
        if device == "cpu":
            return device
        raise ValueError(device)

    monkeypatch.setattr(paddle_utils.paddle, "set_device", fake_set_device)

    assert paddle_utils.resolve_runtime_device("auto") == "cpu"
    assert calls == ["gpu", "xpu", "cpu"]


def test_move_to_device_supports_tensor_and_layer():
    x = paddle.randn([2, 3])
    layer = paddle.nn.Linear(3, 4)

    x_cpu = paddle_utils.move_to_device(x, "cpu")
    layer_cpu = paddle_utils.move_to_device(layer, "cpu")
    weight = layer_cpu.parameters()[0]

    assert str(x_cpu.place) == "Place(cpu)"
    assert str(weight.place) == "Place(cpu)"
    assert layer_cpu is layer


def test_no_hardcoded_cuda_calls_remain():
    patterns = [
        ".cuda(",
        "paddle.cuda.manual_seed(",
        'paddle.device("cuda")',
        "paddle.device('cuda')",
    ]

    offenders = []
    for file_path in PADDLE_ROOT.rglob("*.py"):
        content = file_path.read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern in content:
                offenders.append((file_path.relative_to(PROJECT_ROOT), pattern))

    assert offenders == []
