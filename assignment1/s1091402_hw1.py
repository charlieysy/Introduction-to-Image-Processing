import cv2
import numpy as np

img = cv2.imread('yzu.bmp')
cv2.imshow('Rotate image', img)
cv2.imshow('Rotate CR image', img)

(h, w, d) = img.shape
center = (w//2, h//2)
radius = min(w//2, h//2)

def rotate(angle):
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    cv2.imshow('Rotate image', rotated) #update img
    
def rotateCR(angle):
    mask = np.zeros_like(img)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    #cv2.imshow('Mask',mask)

    mask2=cv2.bitwise_not(mask)
    cv2.circle(mask2, center, radius, (0, 0, 0), -1)
    #cv2.imshow('Mask2',mask2)
    
    cr=cv2.bitwise_and(img,mask)
    #cv2.imshow('cr',cr)
    background=cv2.bitwise_and(img,mask2)
    #cv2.imshow('background',background)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(cr, M, (w, h))
    result=cv2.add(background,rotated)
    cv2.imshow('Rotate CR image',result)

cv2.createTrackbar('degree', 'Rotate image', 0, 359, rotate)
cv2.createTrackbar('degree', 'Rotate CR image', 0, 359, rotateCR)


cv2.waitKey(0)
cv2.destroyAllWindows()

