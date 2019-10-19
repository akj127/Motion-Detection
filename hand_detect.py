import cv2
import numpy as np
img = cv2.imread("hand.jpg", -1)
print(img)
cv2.imshow("hand",img)
cv2.waitKey(0)
