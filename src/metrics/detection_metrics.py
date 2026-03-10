"""Compact detection metrics for the baseline detector pipeline."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors or sequences to numpy arrays."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU for XYXY boxes."""
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    boxes1 = boxes1.astype(np.float32)
    boxes2 = boxes2.astype(np.float32)

    top_left = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(bottom_right - top_left, a_min=0.0, a_max=None)
    intersection = wh[..., 0] * wh[..., 1]

    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0.0, None) * np.clip(boxes1[:, 3] - boxes1[:, 1], 0.0, None)
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0.0, None) * np.clip(boxes2[:, 3] - boxes2[:, 1], 0.0, None)
    union = np.clip(area1[:, None] + area2[None, :] - intersection, a_min=1e-8, a_max=None)
    return intersection / union


def _compute_average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP by integrating the precision-recall curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])

    changing_points = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1]))


def _evaluate_class_at_iou(
    predictions: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    class_id: int,
    iou_threshold: float,
    score_threshold: float,
) -> Dict[str, float]:
    gt_by_image: Dict[str, np.ndarray] = {}
    matched_gt: Dict[str, np.ndarray] = {}
    num_gt = 0

    for target in targets:
        target_labels = _to_numpy(target["labels"]).astype(np.int64)
        target_boxes = _to_numpy(target["boxes"]).astype(np.float32)
        class_boxes = target_boxes[target_labels == class_id]
        gt_by_image[target["image_id"]] = class_boxes
        matched_gt[target["image_id"]] = np.zeros((len(class_boxes),), dtype=bool)
        num_gt += len(class_boxes)

    prediction_rows: list[tuple[str, float, np.ndarray]] = []
    for prediction in predictions:
        pred_labels = _to_numpy(prediction["labels"]).astype(np.int64)
        pred_scores = _to_numpy(prediction["scores"]).astype(np.float32)
        pred_boxes = _to_numpy(prediction["boxes"]).astype(np.float32)
        keep = (pred_labels == class_id) & (pred_scores >= score_threshold)
        kept_scores = pred_scores[keep]
        kept_boxes = pred_boxes[keep]
        for score, box in zip(kept_scores, kept_boxes):
            prediction_rows.append((prediction["image_id"], float(score), box))

    prediction_rows.sort(key=lambda item: item[1], reverse=True)
    if not prediction_rows:
        return {
            "ap": float("nan") if num_gt == 0 else 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": float(num_gt),
            "num_gt": float(num_gt),
            "num_predictions": 0.0,
        }

    tp = np.zeros((len(prediction_rows),), dtype=np.float32)
    fp = np.zeros((len(prediction_rows),), dtype=np.float32)

    for index, (image_id, _, pred_box) in enumerate(prediction_rows):
        gt_boxes = gt_by_image.get(image_id, np.zeros((0, 4), dtype=np.float32))
        if gt_boxes.size == 0:
            fp[index] = 1.0
            continue

        ious = box_iou_numpy(pred_box[None, :], gt_boxes)[0]
        max_iou_index = int(np.argmax(ious))
        max_iou = float(ious[max_iou_index])

        if max_iou >= iou_threshold and not matched_gt[image_id][max_iou_index]:
            matched_gt[image_id][max_iou_index] = True
            tp[index] = 1.0
        else:
            fp[index] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision_curve = tp_cum / np.clip(tp_cum + fp_cum, a_min=1e-8, a_max=None)
    recall_curve = tp_cum / max(float(num_gt), 1e-8)
    ap = _compute_average_precision(recall_curve, precision_curve) if num_gt > 0 else float("nan")

    tp_total = float(tp.sum())
    fp_total = float(fp.sum())
    fn_total = float(max(num_gt - tp_total, 0.0))
    precision = tp_total / max(tp_total + fp_total, 1e-8)
    recall = tp_total / max(float(num_gt), 1e-8) if num_gt > 0 else 0.0

    return {
        "ap": ap,
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "num_gt": float(num_gt),
        "num_predictions": float(len(prediction_rows)),
    }


def compute_detection_metrics(
    predictions: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    score_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Compute compact AP-style metrics without external evaluation packages."""
    if not class_names:
        raise ValueError("class_names is required to compute detection metrics.")

    iou_thresholds = [round(threshold, 2) for threshold in np.arange(0.50, 1.00, 0.05)]
    per_class_rows: list[Dict[str, Any]] = []

    for class_id, class_name in enumerate(class_names):
        results_per_iou = {
            iou: _evaluate_class_at_iou(
                predictions=predictions,
                targets=targets,
                class_id=class_id,
                iou_threshold=iou,
                score_threshold=score_threshold,
            )
            for iou in iou_thresholds
        }

        ap_values = [result["ap"] for result in results_per_iou.values() if not np.isnan(result["ap"])]
        ap50_result = results_per_iou[0.5]
        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_count": int(ap50_result["num_gt"]),
                "prediction_count": int(ap50_result["num_predictions"]),
                "ap50": round(float(ap50_result["ap"]), 6) if not np.isnan(ap50_result["ap"]) else np.nan,
                "ap50_95": round(float(np.mean(ap_values)), 6) if ap_values else np.nan,
                "precision50": round(float(ap50_result["precision"]), 6),
                "recall50": round(float(ap50_result["recall"]), 6),
                "tp50": int(ap50_result["tp"]),
                "fp50": int(ap50_result["fp"]),
                "fn50": int(ap50_result["fn"]),
            }
        )

    per_class_df = pd.DataFrame(per_class_rows)
    valid_ap50 = per_class_df["ap50"].dropna()
    valid_ap50_95 = per_class_df["ap50_95"].dropna()

    tp_total = int(per_class_df["tp50"].sum()) if not per_class_df.empty else 0
    fp_total = int(per_class_df["fp50"].sum()) if not per_class_df.empty else 0
    fn_total = int(per_class_df["fn50"].sum()) if not per_class_df.empty else 0

    metrics = {
        "mAP50": round(float(valid_ap50.mean()), 6) if not valid_ap50.empty else 0.0,
        "mAP50_95": round(float(valid_ap50_95.mean()), 6) if not valid_ap50_95.empty else 0.0,
        "precision50": round(float(tp_total / max(tp_total + fp_total, 1)), 6),
        "recall50": round(float(tp_total / max(tp_total + fn_total, 1)), 6),
        "num_images": int(len(targets)),
        "num_predictions": int(sum(len(_to_numpy(prediction["scores"])) for prediction in predictions)),
        "num_targets": int(sum(len(_to_numpy(target["labels"])) for target in targets)),
    }

    return {
        "summary": metrics,
        "per_class": per_class_df,
    }
