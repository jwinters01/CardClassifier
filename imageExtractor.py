import cv2
import numpy as np
import imutils
import glob

img_rgb = cv2.imread('Data/test-images/test.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('Data/Images/Ixalan/Air Elemental.jpg', 0)
template = imutils.resize(template, width = 125)
template = cv2.Canny(template, 200, 200)
cv2.imshow('test',template)
(tH, tW) = template.shape[:2]
cv2.imshow('Template', template)
cv2.waitKey(0)
cv2.destroyAllWindows()

# loop over different possible scales
found = None
for scale in np.linspace(0.2,1.0,20)[::-1]:
	# resize the image, keep track of ration of resizing
	resized = imutils.resize(img_gray, width = int(img_gray.shape[1]*scale))
	r = img_gray.shape[1]/ float(resized.shape[1])
	
	# if resized image is smaller than the template, break from the loop.
	if resized.shape[0] < tH or resized.shape[1] < tW:
		break
	#detect edges in the resized image and apply template matching
	edged = cv2.Canny(resized, 200, 200)
	result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
	(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
	
	# draw a bounding box around the detected region
	clone = np.dstack([edged, edged, edged])
	cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0]+tW, maxLoc[1]+tH),
		(0,0,255), 2)
	cv2.imshow("visualize", clone)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	#if we found a new maximum correlation value, then update
	if found is None or maxVal > found[0]:
		found = (maxVal, maxLoc, r)
	
# unpack found, and compute bounding box
(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0]*r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0]+tW)*r), int((maxLoc[1]+tH)*r))

# draw bounding box around detected result
cv2.rectangle(img_rgb, (startX, startY), (endX, endY), (0,0,255), 2)
	

cv2.imshow('Detected',img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()