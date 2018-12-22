import numpy as np
import cv2 as cv
import argparse
from deep_text_detect import text_detect, text_spotter
from read_license import read

parser = argparse.ArgumentParser()
parser.add_argument("image", help="Path to car image")
args = parser.parse_args()
image_path = args.image


image = cv.imread(image_path)
original = image.copy()
cur, coords = text_detect(image, text_spotter)

if cur is not None:
	cur = cv.resize(cur, (600, 200), None, interpolation = cv.INTER_LINEAR)
	plate, pic = read(cur)

	if len(plate) > 5:
		left, top, width, height = coords
		cv.rectangle(image, (left, top), (left + width, top + height), (0, 128, 0), 2)
		cv.imshow("plate", image)
		cv.waitKey(0)
		print(plate)
		quit()
		

image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 30))
opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

image_sub = cv.subtract(image, opening) #same as Tophat

retval, thresh1 = cv.threshold(image_sub, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

output = cv.connectedComponentsWithStats(thresh1, 4, cv.CV_32S)

stats = output[2]

stats = sorted(stats, key = lambda x : x[4], reverse = True)[1:16]
stats = sorted(stats, key = lambda x : x[1], reverse = True)

for i in range(len(stats)):
	left, top, width, height, area = stats[i]
	
	if area > 50 and width / height > 1.5 and width/height < 6:			
		box = original[top : top + height, left :left + width]
		box = cv.resize(box, (600, 200), None, interpolation = cv.INTER_LINEAR)
		cv.imshow("plate", box)
		cv.waitKey(0)
		plate, pic = read(box)

		if len(plate) > 2:
			cv.rectangle(original, (left, top), (left + width, top + height), (0, 128, 0), 2)
			cv.imshow("plate", original)
			cv.waitKey(0)
			print(plate)
			quit()


	# quit()

