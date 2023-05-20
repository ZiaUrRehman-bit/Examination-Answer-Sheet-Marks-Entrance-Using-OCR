import cv2
print(cv2.__version__)
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

import os
import numpy as np

imgq = cv2.imread("QueryImage.PNG")
h,w,c = imgq.shape
# imgq = cv2.resize(imgq, (w//2, h//2))

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgq,None)
# imgKp1 = cv2.drawKeypoints(imgq, kp1, None)
per = 25

roi = []

path = 'UserForms'
myImageList = os.listdir(path)
print(myImageList)

for j, y in enumerate(myImageList):
    img = cv2.imread(path+"/"+y)
    # img = cv2.resize(img, (w,h))
    # cv2.imshow("ss", img)
    kp2, des2 =  orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    # matches.sort(key= lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]

    imgMatch = cv2.drawMatches(img, kp2, imgq, kp1, good, None, flags=2)
    # imgMatch = cv2.resize(imgMatch, (w,h))
    # cv2.imshow(y, imgMatch)

    srcPonits = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPonits = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPonits, dstPonits, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))
    # imgScan = cv2.resize(imgScan, (w,h))
    cv2.imshow(y, imgScan)



# cv2.imshow("Image KP1", imgKp1)
cv2.imshow("Image", imgq)
cv2.waitKey()
cv2.destroyAllWindows()