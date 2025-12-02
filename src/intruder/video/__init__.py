import numpy as np
from typing import Literal

CocoBbox = tuple[float, float, float ,float] #[x_tl,y_tl,w,h]
HIGHT = int
WIDTH = int
FRAME = np.ndarray[tuple[HIGHT,WIDTH],np.dtype[np.uint8]]
GVIDEO = np.ndarray[tuple[int,HIGHT,WIDTH],np.dtype[np.uint8]]
RGBVIDEO = np.ndarray[tuple[Literal[3],HIGHT,WIDTH],np.dtype[np.uint8]]
