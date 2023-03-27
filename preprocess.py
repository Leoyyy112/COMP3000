import numpy as np
import cv2

filename = '/Users/liuyang/IROS-Segmentation/demo1.jpeg'

image = cv2.imread(filename)
print(np.shape(image))
print(image.dtype)
image = cv2.resize(image,(256, 256))
print(np.shape(image))