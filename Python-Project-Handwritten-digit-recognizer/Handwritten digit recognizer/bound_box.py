import cv2
import os
import numpy as np

img = cv2.imread('box_expression_5.jpg', 0) 
path = r"jpg_photos"

cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)
image, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
sorted_ctrs = sorted(contours, key = lambda ctr: cv2.boundingRect(ctr)[0])
ROI_number = 0

for c in sorted_ctrs:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    ROI = img[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(path, 'ROI_{}.jpg'.format(ROI_number)), ROI)
    # draw a white rectangle to visualize the bounding rect
    cv2.rectangle(img, (x, y), (x + w, y + h), 255, 1)
    ROI_number += 1

cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
cv2.imwrite("output_box_image.jpg",img)

