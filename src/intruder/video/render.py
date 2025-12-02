from typing import Union
import numpy as np
import cv2
from . import GVIDEO, RGBVIDEO
from .video import VideoGeneratedSample

Color = Union[int, tuple[int, int, int], list[int]]


def render_detections(
    sample:VideoGeneratedSample,
    *,
    color: Color = (0, 255, 0),
    thickness: int = 2,
    return_rgb: bool = False,
) -> GVIDEO | RGBVIDEO:
    # 1) Copy video and ensure dtype uint8
    video_in = sample.video
    if video_in.dtype != np.uint8:
        raise ValueError(f"Expected sample.video dtype uint8, got {video_in.dtype}")

    video_out = video_in.copy()  # (T, H, W)

    # 2) Expand to 3 channels if needed (channel-first: (T,3,H,W))
    if return_rgb:
        # Repeat along a new channel axis
        video_out = np.repeat(video_out[:, None, :, :], 3, axis=1)  # (T,3,H,W)

    num_frames = video_out.shape[0]

    for frame_idx in range(num_frames):
        # Select frame in the shape OpenCV expects
        if return_rgb:
            # (3,H,W) -> (H,W,3), then ensure contiguity
            frame = video_out[frame_idx].transpose(1, 2, 0)
            frame = np.ascontiguousarray(frame)
        else:
            # (H,W) grayscale; ensure contiguity
            frame = video_out[frame_idx]
            frame = np.ascontiguousarray(frame)

        # Get detections for this frame
        bboxes = sample.labels.video_detections.get(frame_idx, [])

        # Prepare color in BGR or grayscale
        if return_rgb:
            if isinstance(color, int):
                cv_color = (int(color), int(color), int(color))
            else:
                # Provided as RGB -> convert to BGR for OpenCV
                r, g, b = map(int, color)
                cv_color = (b, g, r)
        else:
            # Grayscale expects a single int
            cv_color = int(color) if isinstance(color, int) else int(color[0])

        # Draw rectangles
        H, W = frame.shape[:2]
        for x, y, w, h in bboxes:
            x1 = max(0, min(int(round(x)), W - 1))
            y1 = max(0, min(int(round(y)), H - 1))
            x2 = max(0, min(int(round(x + w)), W - 1))
            y2 = max(0, min(int(round(y + h)), H - 1))
            # Ensure x2 >= x1 and y2 >= y1 to avoid errors
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), cv_color, thickness)

        # Write back
        if return_rgb:
            # (H,W,3) -> (3,H,W)
            video_out[frame_idx] = frame.transpose(2, 0, 1)
        else:
            video_out[frame_idx] = frame

    # Return as-is (T,3,H,W) for RGB, (T,H,W) for gray
    return video_out
