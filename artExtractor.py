import cv2
TEST_IMG= "data/Images/Ixalan/Deadeye Tormentor.jpg"


#reading the image 
image = cv2.imread(TEST_IMG)
edged = cv2.Canny(image, 250, 250)
cv2.imshow("Edges", edged)
cv2.waitKey(0)

##### CAVEOFPROGRAMMING THRESHOLDING ####
# setting threshold
retval, threshold = cv2.threshold(image, 12, 255, cv2.THRESH_BINARY)
cv2.imshow('original', image)
cv2.imshow('threshold', threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

# to grayscaled
grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
retval, threshold = cv2.threshold(grayscaled, 10, 255, cv2.THRESH_BINARY)
cv2.imshow('original', image)
cv2.imshow('threshold', threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

# thresholding
th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,105,1)
cv2.imshow('original', image)
cv2.imshow('Adaptive threshold', th)
cv2.waitKey(0)
cv2.destroyAllWindows()

#applying the closing function
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closed", closed)
cv2.waitKey(0)
cv2.destroyAllWindows()

#finding contours
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02*peri, True)
	cv2.drawContours(image, [approx], -1, (180,255,75), 2)
cv2.imshow("Output", image)
cv2.waitKey(0)