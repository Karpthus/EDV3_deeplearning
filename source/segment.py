import os
import glob
import cv2 as cv
import numpy as np

def maskPurpleBG(img):
    """ Asssuming the background is blue, segment the image and return a
        BW image with foreground (white) and background (black)
    """ 
    # Change image color space
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    light_purple = (35,0,35)  # converted from HSV value obtained with colorpicker (150,50,0)
    dark_purple  = (250,255,255)  # converted from HSV value obtained with colorpicker (250,100,100)

    mask = ~cv.inRange(img_hsv, light_purple, dark_purple)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((7,7),np.uint8), mask)
    # cv.imshow("Mask", mask)
    return mask


if __name__ == "__main__":
    """ Test segmentation functions"""
    #data_path = r'C:\GoogleDrive\U-shaped assambly cell\SmartBeamerV2\data\TwoEuro'
    #data_path = r'C:\GoogleDrive\U-shaped assambly cell\SmartBeamerV2\data\OneEuro'
    data_path = r'C:\Users\stijn\Pictures\deeplearning\FiveCent'

    # grab the list of images in our data directory
    print("[INFO] loading images...")
    p = os.path.sep.join([data_path, '**', '*.png'])

    file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
    print("[INFO] images found: {}".format(len(file_list)))

    # loop over the image paths
    for filename in file_list:
        
        # load image and blur a bit
        img = cv.imread(filename)
        #img = cv.blur(img,(3,3))        

        # mask background 
        mask = maskPurpleBG(img)
        masked_img = cv.bitwise_and(img, img, mask=mask)

        # show result and wait a bit        
        cv.imshow("Masked image", masked_img)
        k = cv.waitKey(1000) & 0xFF

        # if the `q` key or ESC was pressed, break from the loop
        if k == ord("q") or k == 27:
            break 
    
