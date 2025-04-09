# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:23:17 2021

@author: ChoongminKim
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

thresh_hold = 200
ratio = 0.7

def segmentation(img_original):
    img = img_original.copy()
    
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.normalize(img2, None, 0, img2.ravel().max(), cv2.NORM_MINMAX)
    
    img_max_value = np.uint8(img2.ravel().max()*ratio)
    if img_max_value > thresh_hold:
        value = img_max_value
    else:
        value = thresh_hold
    
    ret, thresh = cv2.threshold(img2, value, 255, cv2.THRESH_BINARY)
    
    # noise removal
    kernel = np.ones((3, 3),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel, iterations=1)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=1)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    
    ret, sure_fg = cv2.threshold(dist_transform,0.15*dist_transform.max(),255, cv2.THRESH_BINARY)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    #unknown = cv2.subtract(sure_bg,sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [0, 200, 200]
    
    #return img, cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2BGR), cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2BGR), cv2.cvtColor(unknown, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
    path = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(path, 'img')
    data_set = os.listdir(data_path)
    data_set = [data for data in data_set if data.endswith('.jpg')]
    
    for data in data_set:
        img = cv2.imread(os.path.join(data_path, data), cv2.IMREAD_UNCHANGED)
        img_seg = image_segmentation(img)
        add_img = np.hstack([img, img_seg])
        
        #img_seg, temp, temp1, temp2 = image_segmentation(img)
        #add_img = np.hstack([img, img_seg, temp, temp1, temp2])
        
        cv2.imwrite(f'./segmentation/segmentation_{data}.jpg', add_img)

if __name__ == "__main__":
    main()









