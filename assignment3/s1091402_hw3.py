import cv2
import numpy as np

img = cv2.imread('image3.png', cv2.IMREAD_GRAYSCALE)

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

# compute phase spectrum
phase = cv2.phase(planes[0], planes[1], angleInDegrees=True)
phase = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# compute inverse DFT
idft = cv2.idft(DFT)
idft = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
idft = cv2.normalize(idft, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow('Spectrum', S)
cv2.imshow('phase', phase)
cv2.imshow('IDFT', idft)
cv2.waitKey(0)
cv2.destroyAllWindows()
