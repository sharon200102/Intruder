import numpy as np


CocoBbox = tuple[float, float, float ,float] #[x_tl,y_tl,w,h]
HIGHT = int
WIDTH = int
FRAME = np.ndarray[tuple[HIGHT,WIDTH],np.dtype[np.uint8]]