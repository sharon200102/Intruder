import abc
from . import FRAME
import cv2
class BackgroundGenerator(abc.ABC):

    @abc.abstractmethod
    def render_background(self, frame:FRAME) -> FRAME:
        raise NotImplementedError('Abstract')


class GridBackgroundGenerator(BackgroundGenerator):

    def __init__(self,rows:int,cols:int,color:int=255,thickness:int=10):
        super().__init__()
        self._rows = rows
        self._cols = cols
        self._color = color
        self._thickness = thickness

    
    def render_background(self, frame:FRAME) -> FRAME:
        height, width = frame.shape[:2]
    
        # Horizontal lines
        for i in range(self._rows + 1):
            y = int(i * height / self._rows)
            cv2.line(frame, (0, y), (width, y), self._color, self._thickness)
        
        # Vertical lines
        for i in range(self._cols + 1):
            x = int(i * width / self._cols)
            cv2.line(frame, (x, 0), (x, height), self._color, self._thickness)
        
        return frame