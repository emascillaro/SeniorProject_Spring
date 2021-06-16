import cv2
import os
import numpy as np

# From Original Code in Github

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


# Old but working code
'''
def box():
    rectangles = [] # NC
    box_heights = []
    img = cv2.imread('box_expression_2.jpg', 0)                                                                    #return to Capture_Image 
    path = r"final_images"

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
        ROI = img[y-15:y+h+15, x-15:x+w+15]
        cv2.imwrite(os.path.join(path, 'ROI_{}.jpg'.format(ROI_number)), ROI)
        # draw a white rectangle to visualize the bounding rect
        cv2.rectangle(img, (x-15, y-15), (x + w + 15, y + h + 15), 255, 1)
        ROI_number += 1

    print(box_heights)
    rectangles, weights = cv2.groupRectangles(rectangles, 1, 1)     # you need to play around with the last value / NC
    print(rectangles) # NC
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    cv2.imwrite("output_box_image.jpg",img)

    return box_heights

box()

'''


'''

img = cv2.imread('box_expression_4.jpg', 0) 
path = r"jpg_photos"

cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)
image, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
sorted_ctrs = sorted(contours, key = lambda ctr: cv2.boundingRect(ctr)[0])
ROI_number = 0

concat = np.concatenate(sorted_ctrs)
print(concat)
hulls = cv2.convexHull(concat)

bbox_list = []
rect_used = []
end_box_list = []
x_pixel_value = 5
y_pixel_value = 6

#print(sorted_ctrs)

for c in sorted_ctrs:
    # get the bounding rect corner points and dimensions
    x, y, w, h = cv2.boundingRect(c)
    bbox_list.append([x,y,x+w,y+h])

for i in bbox_list:
    rect_used.append(False)

print(bbox_list)

for enum, i in enumerate(bbox_list):
    if rect_used[enum] == True:
        continue 
    xmin = i[0]
    ymin = i[1]
    xmax = i[2]
    ymax = i[3]
    #print(xmin, ymin, xmax, ymax)

    for enum1,j in enumerate(bbox_list[(enum+1):], start = (enum+1)):
        i_xmin = j[0]
        i_ymin = j[1]
        i_xmax = j[2]
        i_ymax = j[3]
        #print(i_xmin, i_ymin, i_xmax, i_ymax)

        if rect_used[enum1] == False:
            if abs(ymin - i_ymin) < x_pixel_value:
                if abs(xmin - i_xmax) < y_pixel_value or abs(xmax - i_xmin) < y_pixel_value:
                    rect_used[enum1] = True
                    xmin = min(xmin, i_xmin)
                    ymin = min(ymin, i_ymin)
                    xmax = max(xmax, i_xmax)
                    ymax = max(ymax, i_ymax)
    final_box = [xmin, ymin, xmax, ymax]
    end_box_list.append(final_box)
#print(end_box_list)

for c in end_box_list:
    # get the bounding rect
    ROI = img[i[1]:i[3], i[0]:i[2]]
    cv2.imwrite(os.path.join(path, 'ROI_{}.jpg'.format(ROI_number)), ROI)
    # draw a white rectangle to visualize the bounding rect
    cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), 255, 1)
    ROI_number += 1

cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
cv2.imwrite("output_box_image.jpg",img)

'''


