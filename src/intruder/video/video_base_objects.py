import random
import numpy as np
from pathlib import Path
import dataclasses
from . import CocoBbox
#TODO transform bbox to pydantic class with multiple tests
#TODO transfrom the VideoDetections class to pydantic 

_FILE_DELIMETER = ','
@dataclasses.dataclass
class VideoDetections():
    video_detections:dict[int,list[CocoBbox]] = dataclasses.field(default_factory=dict) # frame level annotations


    @property
    def frames_num(self) -> int:
        return len(self.video_detections)
    
    def add_detection(self,frame_i:int,detection:CocoBbox):
        self.video_detections.setdefault(frame_i,[]).append(detection)

    @classmethod
    def from_mot_file(cls, filepath:Path) -> "VideoDetections":
        video_detections:dict[int,list[CocoBbox]] = {}
        if not filepath.exists():
            raise ValueError(f'File {filepath} does not exist')
        with open(filepath,mode='r',encoding='utf-8') as detections_file:
            for line in detections_file:
                frame_num, x_tl, y_tl, w, h = line.strip().split(_FILE_DELIMETER)
                bbox = tuple(float(element) for element in (x_tl, y_tl, w, h))
                video_detections.setdefault(int(frame_num),[]).append(bbox)
        
        return cls(video_detections)
    
    def to_mot_file(self,filepath:Path):
        """
        Saves detections in simplified MOT format:
        frame,x_tl,y_tl,width,height
        One detection per line, sorted by frame number.
        """
        # Sort by frame index for clean output
        sorted_frames = sorted(self.video_detections.items())

        with open(filepath, mode='w', encoding='utf-8') as f:
            for frame_idx, bboxes in sorted_frames:
                for x_tl, y_tl, w, h in bboxes:
                    # Write with reasonable precision (e.g. 6 decimals)
                    line = f"{frame_idx},{x_tl:.6f},{y_tl:.6f},{w:.6f},{h:.6f}\n"
                    f.write(line)


@dataclasses.dataclass
class BlobObject:
    center: tuple[int,int]
    radius: int
    vx:float
    vy: float

    @classmethod
    def create_random_blob(cls,
                           max_x:int,
                           max_y:int,
                           radius_range:tuple[int,int],
                           speed_range:tuple[float,float]|None = None) -> "BlobObject":
        
        radius = random.randint(radius_range[0], radius_range[1])
        x = random.randint(radius, max_x - radius)
        y = random.randint(radius, max_y - radius)
        if speed_range is None:
            vx = 0
            vy = 0
        else:
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(speed_range[0],speed_range[1])
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)

        return cls(center = (x,y), radius=radius, vx=vx, vy=vy)
    

    def move(self) -> "BlobObject":
        new_x, new_y = (int(self.center[0] + self.vx), int(self.center[1] + self.vy))
        # Small random Brownian-like perturbation
        new_vx = self.vx + random.uniform(-0.4, 0.4)
        new_vy = self.vy + random.uniform(-0.4, 0.4)
        return BlobObject(center=(new_x,new_y),radius=self.radius,vx=new_vx,vy=new_vy)

    



    @property
    def bbox(self) -> CocoBbox:
        tl_x, tl_y = (self.center[0] - self.radius, self.center[1] - self.radius)
        w,h = (self.radius * 2, self.radius * 2)
        return (tl_x, tl_y, w, h)

