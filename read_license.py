import cv2 as cv
import numpy as np
from sklearn.externals import joblib

clf = joblib.load("symbol_detection_random_forest_v1.joblib")  

symbols = "0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z".split(",")
kernel = np.ones((1, 3),np.uint8)
letter_indices = [0, 1, 5, 6]


def get_plate_symbols ( stats, img, vis):
	"""
		Detect symbol with classifier. 
	"""
	plate = ""
	for i in range(len(stats)):
		if len(plate) == 7: break

		left, top, width, height, area = stats[i]
		
		if area < 700 or area > 8000 or width > height or left == 0: continue ## only those regions, more liked to be characters

		cur = img[top:top + height, left:left + width]

		symbol = cv.resize(cur, (100, 35))
		symbol = symbol.reshape(1, -1)
		
		probs = clf.predict_proba(symbol)[0]
		
		if max(probs) < 0.1:
			continue

		digit = clf.predict(symbol)[0] 
		plate += str(symbols[digit])
		
		cv.rectangle(vis, (left, top), (left + width, top + height), (0, 128, 0), 2)
	return plate, vis

def read(img):

	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	vis = img.copy()
	
	retval, img_thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
	
	if retval > 60:
		img_thresh = cv.erode(img_thresh, kernel, iterations = 1)  # remove black spots from image, to seperate characters from each other
	
	img = cv.bitwise_not(img_thresh)
	
	output = cv.connectedComponentsWithStats(img, 4, cv.CV_32S)
	stats = output[2]
	stats = sorted(stats, key = lambda x : x[0])
	plate, vis = get_plate_symbols(stats, img, vis)

	cv.imshow("plate", vis)
	cv.waitKey(0)


	## since classifier is not acurate as should be check for 0 and O
	if len(plate) == 7:
		for i in letter_indices:
			if plate[i] == "0": plate = plate[:i] + "O" + plate[i + 1:]
	
	return plate, vis

		
	
	
