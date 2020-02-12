# resources
# https://nanonets.com/blog/ocr-with-tesseract/ 


import cv2 
import pytesseract
import numpy as np
import math
from scipy import ndimage

# testing files
scanned_file = 'scan_document.png'
receipt_easy = 'receipt_easy.jpeg'
receipt_hard = 'receipt_hard.jpg'
license_plate = 'license_plate.jpg'

images = ['license_plate.jpg', 'receipt_example.jpg', 'scan_document.png']


# baseline - scanned file
img = cv2.imread(scanned_file)
text = pytesseract.image_to_string(img)

print(text)

# receipt
img = cv2.imread(receipt_hard)
text_initial = pytesseract.image_to_string(img)

print(text) # not so great results

def orientation_correction(img, save_image = False):
    # GrayScale Conversion for the Canny Algorithm  
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # Canny Algorithm for edge detection was developed by John F. Canny not Kennedy!! :)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    # Using Houghlines to detect lines
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    
    # Finding angle of lines in polar coordinates
    angles = []
    for x1, y1, x2, y2 in lines[0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    
    # Getting the median angle
    median_angle = np.median(angles)
    
    # Rotating the image with this median angle
    img_rotated = ndimage.rotate(img, median_angle)
    
    if save_image:
        cv2.imwrite('orientation_corrected.jpg', img_rotated)
    return img_rotated

img_rotated = orientation_correction(img)

# preprocessing
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


grey = get_grayscale(img_rotated)
tresh = thresholding(grey)

text_initial = pytesseract.image_to_string(img)
text_rotated = pytesseract.image_to_string(img_rotated)
text_rotated_grey = pytesseract.image_to_string(grey)

print(text_initial)
print(text_rotated)
print(text_rotated_grey)


