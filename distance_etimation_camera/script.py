import cv2 as cv
import numpy as np


cam = cv.VideoCapture(0)
font = cv.FONT_HERSHEY_COMPLEX
kernel = (5,5)

while cam.isOpened():
	isTrue, original_frame = cam.read()
	if isTrue:
		original_frame = cv.flip(original_frame, 1)
		img = cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY)
		img = cv.GaussianBlur(img, kernel, 0)
		img = cv.Canny(img, 75, 175)
		if cv.waitKey(20) & 0xFF == ord('q'):
			break
		cv.imshow("Video1", img)
		img2 = cv.applyColorMap(original_frame, cv.COLORMAP_MAGMA)
		cv.imshow("Video2", img2)
	else:
		cam.realease()
		cv.destroyAllWindows()
		break