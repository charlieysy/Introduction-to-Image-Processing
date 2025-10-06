import cv2
import numpy as np

img = cv2.imread('image4.png', cv2.IMREAD_GRAYSCALE)

P = cv2.getOptimalDFTSize(img.shape[0])
Q = cv2.getOptimalDFTSize(img.shape[1])

# padding
padded_img = cv2.copyMakeBorder(img, 0, P - img.shape[0], 0, Q - img.shape[1], cv2.BORDER_CONSTANT, value=0)
padded_img = padded_img.astype(np.float32)

# center image
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
            padded_img[i][j] *= (-1)**(i+j)

# compute DFT
DFT = cv2.dft(padded_img, flags=cv2.DFT_COMPLEX_OUTPUT)
planes = cv2.split(DFT)
# compute spectrum
S = 20 * np.log(cv2.magnitude(planes[0], planes[1]))
# normalize the output image to [0, 255]
S = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow('S',S)
#以上為作業3的部分

#draw circle
def show_xy(event,x,y,flags,param):
    if event==1:    #mouse click
        cv2.circle(S, (x,y), 7, (0,0,0), -1)    #img,coordinate,radius,color,Solid
        cv2.imshow('S',S)
cv2.setMouseCallback('S', show_xy)

while(1):
    key=cv2.waitKey(0)
    if key==27:
        break
cv2.destroyAllWindows()

S = np.expand_dims(S, axis=2)   #讓S跟fshift形狀一樣(原本一個二維一個三維)
fshift=DFT*S
idft=cv2.idft(fshift, flags=cv2.DFT_COMPLEX_OUTPUT)
planes2 = cv2.split(idft)
idft = cv2.magnitude(planes2[0], planes2[1])
idft = cv2.normalize(idft, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
Result=cv2.GaussianBlur(idft,(11,11),0)

#cv2.imshow('Notch',idft)
cv2.imshow('Result',Result)
cv2.waitKey(0)
cv2.destroyAllWindows()
