import abc
import random
import dataclasses
import cv2
import numpy as np
from typing import Any

from . import GVIDEO

from . import FRAME
from .video_base_objects import BlobObject
from .video_base_objects import VideoDetections

_RETRIES = 5
_RADIUS_RANGE: tuple[int,int] = (3,12)
_SPEED_RANGE: tuple[float,float] = (1.0,3.0)

    


@dataclasses.dataclass
class VideoMetadata:
    width: int
    height: int
    num_seconds: int
    fps:int
    # self._num_frames = self._fps * self._num_seconds
    blobs_color: int
    
    def __post_init__(self) -> None:
        self.num_frames = self.fps * self.num_seconds


@dataclasses.dataclass
class VideoGeneratedSample:
    video: GVIDEO
    labels: VideoDetections

class VideoGenerator(abc.ABC):

    @abc.abstractmethod
    def generate_video(self, video_metadata: VideoMetadata) -> VideoGeneratedSample:
        raise NotImplementedError('abstract')

class WorkShopVideoGenerator(VideoGenerator):
    def __init__(self,num_blobs:int,background_generator:Any = None,should_move:bool=False,
                  can_overlap:bool = False):
        super().__init__()
        self._num_blobs = num_blobs
        self._should_move = should_move
        self._background_generator = background_generator
        self._can_overlap = can_overlap 

    def generate_video(self, video_metadata:VideoMetadata) -> VideoGeneratedSample:
        frames: list[FRAME] = []
        blobs = self._create_inital_blobs(video_metadata)
        video_annotations:VideoDetections = VideoDetections()
        for frame_i in range(video_metadata.num_frames):
            next_state_blobs: list[BlobObject] = []
            frame:FRAME = np.zeros((video_metadata.height, video_metadata.width), dtype=np.uint8)
            if self._background_generator is not None:
                frame = self._render_background(frame)
            for blob in blobs:
                if not self._is_blob_within_frame(blob=blob,
                                                  frame_width=video_metadata.width,
                                                  frame_heigth=video_metadata.height):
                    continue

                self._render_blob(frame=frame,blob=blob,video_metadata=video_metadata)
                video_annotations.add_detection(frame_i,blob.bbox)

                if not self._should_move:
                    next_state_blobs.append(blob)
                else:
                    next_state_blobs.append(blob.move())
            
            frames.append(frame)
            blobs = next_state_blobs
        
        video = np.stack(frames,axis=0)
        return VideoGeneratedSample(video=video,labels=video_annotations)
    

    
    
    def _render_background(self,frame:FRAME) -> FRAME:
        return self._background_generator.render_background(frame)
    
    def _render_blob(self,*,frame:FRAME,blob:BlobObject,video_metadata:VideoMetadata) -> FRAME:
        return cv2.circle(frame, blob.center, blob.radius, video_metadata.blobs_color, -1)
    
    def _is_blobs_intersect(self,blobs:list[BlobObject]):
        n = len(blobs)
        for i in range(n):
            for j in range(i + 1, n):  # avoid checking the same pair twice and self-comparison
                blob_a = blobs[i]
                blob_b = blobs[j]
                
                # Distance between centers
                dx = blob_a.center[0] - blob_b.center[0]
                dy = blob_a.center[1] - blob_b.center[1]
                distance_sq = dx*dx + dy*dy
                
                # Sum of radii squared
                radius_sum = blob_a.radius + blob_b.radius
                if distance_sq <= radius_sum * radius_sum:
                    return True  # intersection found
        
        return False
    

    def _create_inital_blobs(self,video_metadata:VideoMetadata) -> list[BlobObject]:
        
        blobs:list[BlobObject] =  [BlobObject.create_random_blob(video_metadata.width,
                                                     video_metadata.height,_RADIUS_RANGE,
                                                     _SPEED_RANGE) for _ in range(self._num_blobs)]
        if self._can_overlap:
            return blobs
        
        if self._is_blobs_intersect(blobs):
            for _ in range(_RETRIES):
                blobs = [BlobObject.create_random_blob(video_metadata.width,
                                                     video_metadata.height,_RADIUS_RANGE,
                                                     _SPEED_RANGE) for _ in range(self._num_blobs)]
                if not self._is_blobs_intersect(blobs):
                    return blobs
        
        return blobs
    

    
    
    @staticmethod
    def _is_blob_within_frame(*,blob:BlobObject,frame_width:int,frame_heigth:int) -> bool:
        blob_x_tl,blob_y_tl,blob_w,blob_h = blob.bbox
        if any((blob_x_tl < 0,
                blob_y_tl < 0,
                blob_x_tl + blob_w >= frame_width, 
                blob_y_tl + blob_h >= frame_heigth)):
            return False
        return True
    


    



        