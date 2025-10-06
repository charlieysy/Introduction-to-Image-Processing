import cv2

img = cv2.imread('fig.jpg')

IMREAD_GRAYSCALE=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #轉灰階
cv2.imshow('gray',IMREAD_GRAYSCALE)

x = cv2.Sobel(IMREAD_GRAYSCALE, cv2.CV_64F, 1, 0)
y = cv2.Sobel(IMREAD_GRAYSCALE, cv2.CV_64F, 0, 1)

absX = cv2.convertScaleAbs(x) 
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX, 0.5, absX,0.5,0)
cv2.imshow('dst',dst)

sketch=cv2.bitwise_not(dst)
cv2.imshow('sketch',sketch)

cv2.waitKey(0)
cv2.destroyAllWindows()
