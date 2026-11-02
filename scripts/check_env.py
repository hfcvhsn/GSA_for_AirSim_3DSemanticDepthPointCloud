"""Verify imports, pinned source, MobileSAM weights and optional CUDA kernel."""
import argparse
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--cuda", action="store_true", help="Run a real custom CUDA kernel on visible GPU 0")
    parser.add_argument("--models", action="store_true", help="Strictly load the MobileSAM checkpoint on CPU")
    args = parser.parse_args()
    config = load_config(args.config)
    import numpy as np
    import torch
    import torchvision
    import cv2
    import airsim
    import transformers
    import plotly
    source = Path(config["paths"]["gsa_source"])
    for path in (source / "GroundingDINO", source / "segment_anything", source / "EfficientSAM"):
        sys.path.insert(0, str(path))
    from groundingdino import _C
    from src.openuav.client import OpenUAVClient
    client = OpenUAVClient()  # No server connection needed during construction.
    client.client.close()
    assert torch.version.cuda == "11.8", torch.__version__
    assert np.__version__.startswith("1.26."), np.__version__
    assert torch.from_numpy(np.zeros(1)).numpy().shape == (1,)
    expected = json.loads((ROOT / "config/sources.json").read_text())["gsa"]["commit"]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert commit == expected, commit
    record = {"python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "numpy": np.__version__, "torchvision": torchvision.__version__, "opencv": cv2.__version__,
              "transformers": transformers.__version__, "timm": version("timm"), "gsa_commit": commit,
              "extension": _C.__file__, "rpc_adapter_constructed": True}
    if args.models:
        from MobileSAM.setup_mobile_sam import setup_model
        model = setup_model().eval()
        result = model.load_state_dict(torch.load(config["paths"]["sam_checkpoint"], map_location="cpu"), strict=True)
        record.update(mobile_sam_strict_load=str(result), mobile_sam_parameters=sum(p.numel() for p in model.parameters()))
    if args.cuda:
        from groundingdino.models.GroundingDINO.ms_deform_attn import MultiScaleDeformableAttnFunction
        assert torch.cuda.is_available(), "CUDA unavailable"
        value = torch.arange(8, dtype=torch.float32, device="cuda").reshape(1, 4, 1, 2)
        spatial = torch.tensor([[2, 2]], dtype=torch.long, device="cuda")
        starts = torch.tensor([0], dtype=torch.long, device="cuda")
        locations = torch.full((1, 1, 1, 1, 1, 2), .5, device="cuda")
        weights = torch.ones((1, 1, 1, 1, 1), device="cuda")
        actual = MultiScaleDeformableAttnFunction.apply(value, spatial, starts, locations, weights, 64)
        torch.testing.assert_close(actual.cpu(), torch.tensor([[[3., 4.]]]))
        record.update(cuda_kernel="passed", gpu=torch.cuda.get_device_name(0))
    (ROOT / "env/snapshots").mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "env/snapshots/verification.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
