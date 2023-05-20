import cv2

path = "C:\\Users\\hp\\Google Drive\\Fiverr Work\\2022\\37. Examination Answer Sheet Marks Entrance Using OCR\\Updated\\Attendance Sheet Images"
img = cv2.imread(path+"02.jpg")
img = cv2.resize(img, (500,500))
cv2.imshow("Input Image", img)
cv2.waitKey()
cv2.destroyAllWindows()