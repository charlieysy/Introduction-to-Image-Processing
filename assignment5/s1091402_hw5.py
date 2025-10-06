import cv2
import numpy as np

img1 = cv2.imread('img1.jpg')
resized_img1=cv2.resize(img1,(int(img1.shape[1]/10), int(img1.shape[0]/10)))
cv2.imshow('original1',resized_img1)
img2 = cv2.imread('img2.jpg')
resized_img2=cv2.resize(img2,(int(img2.shape[1]//6), int(img2.shape[0]//6)))
cv2.imshow('original2',resized_img2)
img3 = cv2.imread('img3.jpg')
resized_img3=cv2.resize(img3,(int(img3.shape[1]//10), int(img3.shape[0]//10)))
cv2.imshow('original3',resized_img3)

hsv_img1 = cv2.cvtColor(resized_img1, cv2.COLOR_BGR2HSV)
hsv_img2 = cv2.cvtColor(resized_img2, cv2.COLOR_BGR2HSV)
hsv_img3 = cv2.cvtColor(resized_img3, cv2.COLOR_BGR2HSV)

lower_skin = np.array([5, 60, 40], dtype=np.uint8)
upper_skin = np.array([40, 180, 255], dtype=np.uint8)

skin_mask1 = cv2.inRange(hsv_img1, lower_skin, upper_skin)
skin_mask2 = cv2.inRange(hsv_img2, lower_skin, upper_skin)
skin_mask3 = cv2.inRange(hsv_img3, lower_skin, upper_skin)

resized_img1[skin_mask1>0]=(0,0,255)
resized_img2[skin_mask2>0]=(0,0,255)
resized_img3[skin_mask3>0]=(0,0,255)

cv2.imshow('skin Ditection1', resized_img1)
cv2.imshow('skin Ditection2', resized_img2)
cv2.imshow('skin Ditection3', resized_img3)

cv2.imshow('Skin Mask1', skin_mask1)
cv2.imshow('Skin Mask2', skin_mask2)
cv2.imshow('Skin Mask3', skin_mask3)

cv2.waitKey(0)
cv2.destroyAllWindows()