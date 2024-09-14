import numpy as np


saturation_factor = 1.5
brightness_factor = 1.5
blur_kernel_size = (5, 5)
sharpness_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
