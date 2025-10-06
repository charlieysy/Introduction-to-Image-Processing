import cv2
import numpy as np


img = cv2.imread('fig.jpg')
cv2.imshow('img',img)

#img2 = cv2.imread('dst.jpg')
#cv2.imshow('img2',img2)

IMREAD_GRAYSCALE=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('IMREAD_GRAYSCALE',IMREAD_GRAYSCALE)

kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])

kernel_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

gray_x = cv2.filter2D(IMREAD_GRAYSCALE, cv2.CV_64F, kernel_x)
gray_y = cv2.filter2D(IMREAD_GRAYSCALE, cv2.CV_64F, kernel_y)

dst = np.sqrt(gray_x ** 2 + gray_y ** 2)

cv2.imshow('dst',dst)


sketch=cv2.bitwise_not(dst)
cv2.imshow('sketch',sketch)


cv2.waitKey(0)
cv2.destroyAllWindows()
