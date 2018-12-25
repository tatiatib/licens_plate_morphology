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
cur, coords = text_detect(image, text_spotter)  ##SSD text spotter, works for wide shots

max_plate = None
max_dims = None


if cur is not None:
	cur = cv.resize(cur, (600, 200), None, interpolation = cv.INTER_LINEAR)
	plate, pic = read(cur)

	if len(plate) > 4:
		left, top, width, height = coords
		cv.rectangle(image, (left, top), (left + width, top + height), (0, 128, 0), 2)
		cv.imshow("plate", image)
		cv.waitKey(0)
		
		print(plate)
		quit()
	else:
		max_plate = plate
		max_dims = coords
		

image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 4)) # larger than license plate
opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel) #remove salt and increase dark spots

image_sub = cv.subtract(image, opening) #with open - same as Tophat  

retval, thresh1 = cv.threshold(image_sub, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

output = cv.connectedComponentsWithStats(thresh1, 4, cv.CV_32S)

stats = output[2]

stats = sorted(stats, key = lambda x : x[4], reverse = True)[1:16] #largest components
stats = sorted(stats, key = lambda x : x[1], reverse = True) # sort from bottom to the top


for i in range(len(stats)):
	left, top, width, height, area = stats[i]
	
	if area > 50 and width / height > 1.5 and width/height < 6:			
		box = original[top : top + height, left :left + width]
		box = cv.resize(box, (600, 200), None, interpolation = cv.INTER_LINEAR)
		cv.imshow("plate", box)
		cv.waitKey(0)
		plate, pic = read(box)
		
		if len(plate) == 7:
			cv.rectangle(original, (left, top), (left + width, top + height), (0, 128, 0), 2)
			cv.imshow("plate", original)
			cv.waitKey(0)
			
			print(plate)
			quit()
		elif len(plate) > 0 and (max_plate is None  or len(plate) > len(max_plate)): 
			
			max_plate = plate	
			max_dims = (left, top, width, height)

if max_plate:
	left, top, width, height = max_dims
	cv.rectangle(original, (left, top), (left + width, top + height), (0, 128, 0), 2)
	cv.imshow("plate", original)
	cv.waitKey(0)

	print(max_plate)
else:
	print("Nothing found")

