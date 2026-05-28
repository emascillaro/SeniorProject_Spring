# this file creates bounding boxes around digits and operators

import cv2
import os
import numpy as np

def box():
    rectangles = [] # NC
    box_heights = []
    img = cv2.imread('box_expression_3.jpg', 0)          #return to Capture_Image 
    path = r"final_images"
    dt = 10

    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)
    image, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_ctrs = sorted(contours, key = lambda ctr: cv2.boundingRect(ctr)[0])
    ROI_number = 0

    for c in sorted_ctrs:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        box_heights.append(h)
        rect = [x, y, w, h] # NC
        rectangles.append(rect) # NC
        shifted_rect = [x,y+dt,w,h] 
        rectangles.append(shifted_rect) # NC
        intersection = shifted_rect[1] - rect[1]
        if intersection < 0:
            # Join the two rectangles
            rectangles, weights = cv2.groupRectangles(rectangles, 1, 300)     # you need to play around with the last value / NC
            rectangles.remove(rect)
        rectangles.remove(shifted_rect)
        ROI = img[y-10:y+h+10, x-10:x+w+10]
        cv2.imwrite(os.path.join(path, 'ROI_{}.jpg'.format(ROI_number)), ROI)
        # draw a white rectangle to visualize the bounding rect
        cv2.rectangle(img, (x-10, y-10), (x + w + 10, y + h + 10), 255, 1)
        ROI_number += 1

    print(box_heights)
    
    print(rectangles) # NC
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    cv2.imwrite("output_box_image.jpg",img)

    return box_heights

box()
