from pathlib import Path
import numpy as np
import cv2
from typing import Callable
from . import tasks_root_path,TASK_VALID_IDS
from .video import GVIDEO,RGBVIDEO
from .video.video import (
 VideoGenerator,
 WorkShopVideoGenerator,
 VideoMetadata, 
 FRAME,VideoGeneratedSample
)
from .evaluation.evaluate import compute_video_f1_score
from .video.background import GridBackgroundGenerator
from .video.video_base_objects import CocoBbox,VideoDetections
from .video.render import render_detections
class WorkshopRunner:
    _TASK_TO_VIDEO_GENERATOR: dict[int,WorkShopVideoGenerator] = \
    {1: WorkShopVideoGenerator(num_blobs=5),
     2: WorkShopVideoGenerator(num_blobs=5,should_move=True,can_overlap=True),
     3: WorkShopVideoGenerator(num_blobs=5,
                               background_generator=GridBackgroundGenerator(rows=5,cols=5),
                               should_move=True,
                               can_overlap=True)}
    
    _VIDEO_METADATA = VideoMetadata(width=640,height=512,num_seconds=10,fps=5,blobs_color=255)
    _N_GENERATE = 2

    
    def __init__(self,task_num:int):
        if task_num not in TASK_VALID_IDS:
            raise ValueError(f" Task number if invalid {task_num}")

        self._task_num = task_num
        self._task_root_dir = tasks_root_path / f'task_{task_num}'
        self._task_data_dir = self._task_root_dir / 'data'
        self._task_results_dir = self._task_root_dir / 'results'

        self._task_root_dir.mkdir(exist_ok=True,parents=True)
        self._task_data_dir.mkdir(exist_ok=True,parents=True)
        self._example_video_path = self._task_data_dir / 'exmple.mp4'


        self._video_generator:VideoGenerator = type(self)._TASK_TO_VIDEO_GENERATOR[self._task_num]
        self._create_task_data()
    
    def _create_task_data(self,):
        generated_sample = self._video_generator.generate_video(type(self)._VIDEO_METADATA)
        
        self._write_video(generated_sample.video,self._example_video_path)
        print(f"An example video for task {self._task_num} was created in {self._example_video_path}")


    def _write_video(self, frames: GVIDEO | RGBVIDEO, output_path: Path) -> None:
        """
        Write a video to disk using OpenCV.
        Supports:
        - GVIDEO:  (T, H, W) uint8 → grayscale
        - RGBVIDEO: (T, 3, H, W) or (T, H, W, 3) uint8 → color (converted to BGR for OpenCV)
        """
        frames = np.asarray(frames)  # ensure it's a numpy array

        if frames.dtype != np.uint8:
            print("Warning: frames are not uint8. Converting (may clip values).")
            frames = frames.astype(np.uint8)

        if frames.ndim == 3:
            # Grayscale video: (T, H, W)
            num_frames, height, width = frames.shape
            channels = 1
            is_color = False
            frame_for_cv = frames  # (T, H, W)

        elif frames.ndim == 4:
            # Color video: either (T, 3, H, W) or (T, H, W, 3)
            if frames.shape[1] == 3:
                # (T, 3, H, W) → most common in deep learning
                num_frames, _, height, width = frames.shape
                channels = 3
            elif frames.shape[-1] == 3:
                # (T, H, W, 3)
                num_frames, height, width, _ = frames.shape
                channels = 3
            else:
                raise ValueError(f"Invalid 4D shape for RGB video: {frames.shape}. "
                                "Expected (T,3,H,W) or (T,H,W,3)")

            is_color = True
            # Convert RGB → BGR once (OpenCV expects BGR)
            if channels == 3:
                if frames.shape[1] == 3:
                    # (T,3,H,W) → transpose to (T,H,W,3) then BGR
                    frames = frames.transpose(0, 2, 3, 1)  # → (T,H,W,3)
                # Now frames is (T,H,W,3) in RGB order
                frames = frames[..., ::-1]  # RGB → BGR in-place
                frame_for_cv = frames

        else:
            raise ValueError(f"Expected 3D (T,H,W) or 4D (T,3,H,W)/(T,H,W,3) array, got ndim={frames.ndim}")

        fps = type(self)._VIDEO_METADATA.fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=is_color)

        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {output_path}. "
                            "Try different codec (e.g., 'XVID') or file extension.")

        for i in range(num_frames):
            if channels == 1:
                frame = frame_for_cv[i]                  # (H, W)
            else:
                frame = frame_for_cv[i]                  # (H, W, 3) already in BGR
            writer.write(frame)

        writer.release()
        
        
    def run_solution(self,students_solution_fn:Callable[[FRAME],list[CocoBbox]]) -> None:
        try:
            self._task_results_dir.mkdir(exist_ok=True,parents=True)
            video_metadata = type(self)._VIDEO_METADATA
            video_predictions_dict: dict[int,list[CocoBbox]] = {}
            for i in range(type(self)._N_GENERATE):
                generated_sample = self._video_generator.generate_video(video_metadata=video_metadata)

                for frame_idx in range(generated_sample.video.shape[0]):
                    frame = generated_sample.video[frame_idx,:,:].squeeze()
                    try:
                        frame_predictions = students_solution_fn(frame)
                    except Exception as e:
                        raise ValueError(f"Something went wrong, your code crashed while running frame {i} with \
                                          {e}, but don't worry Rome wasn't built in one day either") from e
                    
                    video_predictions_dict[frame_idx] = frame_predictions
            
                video_predictions = VideoDetections(video_predictions_dict)
                predictions_sample = VideoGeneratedSample(video=generated_sample.video,
                                                        labels=video_predictions)
                
                rendered_video = render_detections(predictions_sample,return_rgb=True)
                rendered_video_path = self._task_results_dir/ f'result_{i}.mp4'
                self._write_video(rendered_video,rendered_video_path)
                performance = compute_video_f1_score(generated_sample.labels,video_predictions)
                print(f"Test case {i} your score is {performance.f1} \n")
                print(f"You can view the results of you solution in {self._task_results_dir} \n")
        except Exception as e:
            raise ValueError("Something unpredicted happend, thats ok"
                                f"its probably something that the instructor did not think of {e}") from e
        