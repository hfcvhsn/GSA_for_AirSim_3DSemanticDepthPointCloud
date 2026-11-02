import json

import pytest

from src.config import ROOT, load_config


def test_paths_only_config_inherits_portable_pipeline_settings(tmp_path):
    defaults = json.loads((ROOT / "config/resources.example.json").read_text())
    local = tmp_path / "resource.local.json"
    paths = {key: str(tmp_path / key) for key in defaults["paths"]}
    local.write_text(json.dumps({"paths": paths, "capture": {"port": 42500}}))
    config = load_config(local)
    assert config["paths"] == paths
    assert config["segmentation"] == defaults["segmentation"]
    assert config["cloud"] == defaults["cloud"]
    assert config["capture"]["port"] == 42500
    assert config["capture"]["vehicle_name"] == defaults["capture"]["vehicle_name"]


def test_blank_template_reports_unfilled_fields():
    with pytest.raises(ValueError, match=r"Fill these resource paths.*paths.gsa_source.*paths.cuda_home"):
        load_config(ROOT / "config/resource.local.example.json")


def test_existing_full_config_remains_supported():
    source = json.loads((ROOT / "config/resources.json").read_text())
    config = load_config(ROOT / "config/resources.json")
    assert config["segmentation"] == source["segmentation"]
    assert config["capture"] == source["capture"]
    for key, value in source["paths"].items():
        assert config["paths"][key] == str((ROOT / value).absolute())
