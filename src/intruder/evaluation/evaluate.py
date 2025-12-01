from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional
import numpy as np
from itertools import product
from ..video import CocoBbox
from ..video.video_base_objects import VideoDetections


def bbox_iou(box_a: CocoBbox, box_b: CocoBbox) -> float:
    """Compute IoU between two bounding boxes in (x_tl, y_tl, w, h) format."""
    xa1, ya1, wa, ha = box_a
    xb1, yb1, wb, hb = box_b

    xa2, ya2 = xa1 + wa, ya1 + ha
    xb2, yb2 = xb1 + wb, yb1 + hb

    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)

    inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    box_a_area = wa * ha
    box_b_area = wb * hb
    union_area = box_a_area + box_b_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class FrameMatchingResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0


def match_frame_detections(
    pred_boxes: List[CocoBbox],
    gt_boxes: List[CocoBbox],
    iou_threshold: float = 0.5,
) -> "FrameMatchingResult":
    
    """
    Match predicted boxes to ground truth boxes in a single frame.
    
    Returns counts of TP, FP, FN for this frame.
    """

    if not gt_boxes:
        return FrameMatchingResult(tp=0, fp=len(pred_boxes), fn=0)
    if not pred_boxes:
        return FrameMatchingResult(tp=0, fp=0, fn=len(gt_boxes))

    # Build IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    for pred_i,gt_j in product(range(len(pred_boxes)),range(len(gt_boxes))):
        iou_matrix[pred_i, gt_j] = bbox_iou(pred_boxes[pred_i],gt_boxes[gt_j])

    matched_gt_indices = set()

    # Sort predictions by decreasing max IoU (common heuristic)
    pred_order = np.argsort(-iou_matrix.max(axis=1))
    for pred_idx in pred_order:
        best_gt_idx = np.argmax(iou_matrix[pred_idx])
        if iou_matrix[pred_idx, best_gt_idx] >= iou_threshold and best_gt_idx not in matched_gt_indices:
            matched_gt_indices.add(best_gt_idx)


    tp = len(matched_gt_indices)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return FrameMatchingResult(tp=tp, fp=fp, fn=fn)


@dataclass
class VideoLevelMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "total_gt": self.tp + self.fn,
            "total_pred": self.tp + self.fp,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def __add__(self, other: "VideoLevelMetrics") -> "VideoLevelMetrics":
        return VideoLevelMetrics(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )


def compute_video_f1_score(
    gt: VideoDetections,
    pred: VideoDetections,
    iou_threshold: float = 0.5,
) -> VideoLevelMetrics:
    """
    Compute video-level F1 score by aggregating per-frame matching results.
    
    Highly modular — easy to extend with new matching strategies or filters.
    """
    all_frames = set(gt.video_detections.keys()) | set(pred.video_detections.keys())

    total_metrics = VideoLevelMetrics()

    for frame_idx in sorted(all_frames):
        gt_boxes = gt.video_detections.get(frame_idx, [])
        pred_boxes = pred.video_detections.get(frame_idx, [])

        frame_result = match_frame_detections(
            pred_boxes=pred_boxes,
            gt_boxes=gt_boxes,
            iou_threshold=iou_threshold,
        )

        total_metrics += VideoLevelMetrics(
            tp=frame_result.tp,
            fp=frame_result.fp,
            fn=frame_result.fn,
        )

    return total_metrics