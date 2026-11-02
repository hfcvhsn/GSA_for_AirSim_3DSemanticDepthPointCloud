"""Small adapter around IDEA's pinned Grounded-MobileSAM implementation."""
import importlib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import ROOT, write_json


def validate_classes(classes):
    classes = [name.strip().lower() for name in classes]
    if (not classes or len(classes) >= 65535 or len(set(classes)) != len(classes)
            or any(not name or "." in name for name in classes)):
        raise ValueError("Classes must be nonempty, unique names without periods")
    return classes


def validate_capture(capture):
    """Check registered RGB/depth dimensions before loading model weights."""
    capture = Path(capture)
    metadata = json.loads((capture / "meta.json").read_text())
    for camera, info in metadata["cameras"].items():
        depth = np.load(capture / camera / "depth_perspective_m.npy", mmap_mode="r", allow_pickle=False)
        with Image.open(capture / camera / "rgb.png") as image:
            if depth.shape != (image.height, image.width) or depth.shape != (info["height"], info["width"]):
                raise ValueError(f"Unregistered RGB/depth shape: {camera}")
    return metadata


def segment_capture(capture, output, pipeline, config, max_detections=50, active_pixels=None):
    """Shared five-view stage for the standalone script and end-to-end API."""
    capture, output = Path(capture), Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output is not empty: {output}")
    if not 1 <= max_detections <= 65535:
        raise ValueError("max_detections must be in [1, 65535]")
    metadata = validate_capture(capture)
    colors, counts = palette(len(pipeline.classes) + 1), {}
    output.mkdir(parents=True, exist_ok=True)
    skipped = []
    for camera in metadata["cameras"]:
        active = active_pixels is None or bool(active_pixels[camera].any())
        if active:
            rgb, semantic, instance, confidence, records = pipeline.segment(capture / camera / "rgb.png", max_detections)
        else:
            rgb = np.asarray(Image.open(capture / camera / 'rgb.png').convert('RGB'))
            semantic = np.zeros(rgb.shape[:2], np.uint16)
            instance, confidence, records = semantic.copy(), np.zeros(rgb.shape[:2], np.float32), []
            skipped.append(camera)
        save_segmentation(output / camera, rgb, semantic, instance, confidence, records, colors)
        counts[camera] = len(records)
    report = {
        "schema_version": 1, "capture": str(capture.resolve()),
        "capture_meta_sha256": hashlib.sha256((capture / "meta.json").read_bytes()).hexdigest(),
        "classes": ["unknown"] + pipeline.classes, "colors": colors.tolist(),
        "gsa_commit": json.loads((ROOT / "config/sources.json").read_text())["gsa"]["commit"],
        "settings": {**pipeline.settings, "classes": pipeline.classes}, "max_detections": max_detections,
        "resources": {key: config["paths"][key] for key in ("gsa_source", "dino_checkpoint", "sam_checkpoint", "bert")},
        "detections": counts, "device": pipeline.device,
        "overlap_rule": "highest GroundingDINO score wins", "depth_used_by_segmentation_model": False,
        "depth_used_for_view_selection": active_pixels is not None,
        "skipped_no_action_points": skipped,
    }
    write_json(output / "meta.json", report)
    return report


def palette(count):
    import colorsys
    colors = [[70, 70, 70]]
    colors += [[round(v * 255) for v in colorsys.hsv_to_rgb((i * .61803398875) % 1, .7, .95)]
               for i in range(1, count)]
    return np.array(colors, dtype=np.uint8)


class GroundedMobileSAM:
    def __init__(self, config, classes, device="cuda:0"):
        import torch
        source = Path(config["paths"]["gsa_source"])
        expected = json.loads((ROOT / "config/sources.json").read_text())["gsa"]["commit"]
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise RuntimeError(f"GSA source commit {actual} != pinned {expected}")
        for path in (source, source / "GroundingDINO", source / "segment_anything", source / "EfficientSAM"):
            sys.path.insert(0, str(path))
        # Reuse official image transforms, model loader and original model code.
        self.official = importlib.import_module("grounded_sam_demo")
        from segment_anything import SamPredictor
        from MobileSAM.setup_mobile_sam import setup_model
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; use --device cpu or run outside the restricted sandbox")
            from groundingdino import _C  # Fail early if CUDA extension was not built.
        self.device, self.classes = device, classes
        self.model = self.official.load_model(
            str(source / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
            config["paths"]["dino_checkpoint"], config["paths"]["bert"], device).to(device)
        if config["segmentation"]["sam_model"] != "mobile_sam_tiny_vit":
            raise ValueError("This pipeline uses the official Grounded-MobileSAM TinyViT model")
        mobile_sam = setup_model()
        mobile_sam.load_state_dict(torch.load(config["paths"]["sam_checkpoint"], map_location="cpu"), strict=True)
        self.predictor = SamPredictor(mobile_sam.to(device).eval())
        self.settings = config["segmentation"]
        self.caption = " . ".join(classes) + " ."
        tokenized = self.model.tokenizer(self.caption)
        if len(tokenized["input_ids"]) > 256:
            raise ValueError("Class prompt exceeds GroundingDINO's 256-token limit")
        # Map token spans to a fixed global category table, including multiword names.
        # This avoids assigning IDs from free-form phrases independently per view.
        self.token_groups = []
        start = 0
        for name in classes:
            indices = {tokenized.char_to_token(i) for i in range(start, start + len(name))}
            indices.discard(None)
            if not indices:
                raise ValueError(f"Cannot tokenize category: {name}")
            self.token_groups.append(sorted(indices))
            start += len(name) + 3

    def segment(self, image_path, max_detections=50):
        import torch
        from torchvision.ops import batched_nms, box_convert
        if isinstance(image_path, np.ndarray):
            # Match the official GSA load_image transform without a PNG roundtrip.
            pil = Image.fromarray(image_path)
            T = self.official.T
            transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(),
                                   T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            image, _ = transform(pil, None)
        else:
            pil, image = self.official.load_image(str(image_path))
        rgb = np.array(pil)
        h, w = rgb.shape[:2]
        with torch.inference_mode():
            prediction = self.model(image[None].to(self.device), captions=[self.caption])
            logits = prediction["pred_logits"][0].sigmoid().cpu()
            scores_by_class = torch.stack([logits[:, tokens].max(dim=1).values
                                           for tokens in self.token_groups], dim=1)
            scores, category = scores_by_class.max(dim=1)
            keep = scores > max(self.settings["box_threshold"], self.settings["text_threshold"])
            boxes = box_convert(prediction["pred_boxes"][0].cpu()[keep], "cxcywh", "xyxy")
            boxes *= torch.tensor([w, h, w, h])
            boxes[:, 0::2].clamp_(0, w)
            boxes[:, 1::2].clamp_(0, h)
            scores, category = scores[keep], category[keep]
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, scores, category = boxes[valid], scores[valid], category[valid]
            selected = batched_nms(boxes, scores, category, self.settings["nms_threshold"])[:max_detections]
            boxes, scores, category = boxes[selected], scores[selected], category[selected] + 1
            semantic = np.zeros((h, w), dtype=np.uint16)
            instance = np.zeros((h, w), dtype=np.uint16)
            confidence = np.zeros((h, w), dtype=np.float32)
            records = []
            if len(boxes):
                self.predictor.set_image(rgb)
                transformed = self.predictor.transform.apply_boxes_torch(boxes, (h, w)).to(self.device)
                masks, sam_scores, _ = self.predictor.predict_torch(
                    point_coords=None, point_labels=None, boxes=transformed, multimask_output=True)
                # Match IDEA's MobileSAM demo: choose the highest predicted IoU mask per box.
                best = sam_scores.argmax(dim=1)
                selected_masks = masks[torch.arange(len(masks), device=best.device), best]
                selected_sam_scores = sam_scores[torch.arange(len(masks), device=best.device), best]
                # NMS returns descending scores: the strongest detection owns overlaps.
                for i, (mask, score, label) in enumerate(zip(selected_masks.cpu().numpy(), scores, category), 1):
                    pixels = mask & (instance == 0)
                    semantic[pixels] = int(label)
                    instance[pixels] = i
                    confidence[pixels] = float(score)
                    records.append({"instance_id": i, "class_id": int(label),
                                    "class": self.classes[int(label) - 1], "dino_score": float(score),
                                    "sam_iou_score": float(selected_sam_scores[i - 1]),
                                    "box_xyxy": boxes[i - 1].tolist(), "visible_pixels": int(pixels.sum())})
        return rgb, semantic, instance, confidence, records


def save_segmentation(output, rgb, semantic, instance, confidence, records, colors):
    output.mkdir(parents=True, exist_ok=True)
    for name, array in (("semantic", semantic), ("instances", instance), ("confidence", confidence)):
        np.save(output / f"{name}.npy", array)
    colored = colors[semantic]
    Image.fromarray(colored).save(output / "semantic.png")
    overlay = rgb.copy()
    active = semantic != 0
    overlay[active] = (.5 * rgb[active] + .5 * colored[active]).astype(np.uint8)
    Image.fromarray(overlay).save(output / "overlay.png")
    write_json(output / "segments.json", records)
