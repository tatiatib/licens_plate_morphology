import cv2 as cv
import numpy as np
from sklearn.externals import joblib

clf = joblib.load("symbol_detection_knn_v4.joblib") 

symbols = "0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z".split(",")
kernel = np.ones((1, 3),np.uint8)

def read(img):
	plate = ""
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	vis = img.copy()

	mean = np.mean(img)
	print(mean)
	# retval, img_thresh = cv.threshold(img, mean + 20, 255, cv.THRESH_BINARY)
	retval, img_thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
	
	if retval > 60:
		img_thresh = cv.erode(img_thresh, kernel, iterations = 1)
	
	img = cv.bitwise_not(img_thresh)
	cv.imshow("plate", img_thresh)
	cv.waitKey(0)
	output = cv.connectedComponentsWithStats(img, 4, cv.CV_32S)
	stats = output[2]
	stats = sorted(stats, key = lambda x : x[0])

	for i in range(len(stats)):
		if len(plate) == 7: break

		left, top, width, height, area = stats[i]
		# print(stats[i])
		if area < 600 or area > 7000 or width > height or left == 0: continue

		cur = img[top:top + height, left:left + width]

		symbol = cv.resize(cur, (100, 35))
		symbol = symbol.reshape(1, -1)
		
		digit = clf.predict(symbol)[0] 
		probs = clf.predict_proba(symbol)[0]
		
		if max(probs) < 0.2:
			continue
		plate += str(symbols[digit])
		cv.rectangle(vis, (left, top), (left + width, top + height), (0, 128, 0), 2)
	
	cv.imshow("plate", vis)
	cv.waitKey(0)


	return plate, vis

		
	
	
