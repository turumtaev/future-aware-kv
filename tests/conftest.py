import importlib.util

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        help="Include T=2048 gradient/reference checks",
    )


def pytest_collection_modifyitems(config, items):
    has_cuda = (
        torch.cuda.is_available() and importlib.util.find_spec("triton") is not None
    )
    for item in items:
        if item.get_closest_marker("cuda") and not has_cuda:
            item.add_marker(pytest.mark.skip(reason="CUDA PyTorch and Triton required"))
        elif item.get_closest_marker("slow") and not config.getoption("--run-slow"):
            item.add_marker(
                pytest.mark.skip(reason="Use --run-slow for full-size reference checks")
            )


@pytest.fixture(autouse=True)
def precise_torch_matmul(request):
    """The independent CUDA reference must not silently use TF32 GEMMs."""
    if not request.node.get_closest_marker("cuda"):
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
