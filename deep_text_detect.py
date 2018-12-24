import numpy as np
import cv2 as cv

text_spotter = cv.text.TextDetectorCNN_create("textbox.prototxt", "TextBoxes_icdar13.caffemodel")

def text_detect(img, textSpotter):
    rects, outProbs = textSpotter.detect(img)
    vis = img.copy()
    img_h, img_w,  _ = vis.shape
    
    thresh = 0.5
    done = False

    for r in range(np.shape(rects)[0]):
        if outProbs[r] > thresh:
            x, y, w, h = rects[r]
            start_w = x - 20
            end_w =  x + w + 20
            
            if end_w > img_w: end_w = x + w
            if start_w < 0: start_w = x

            start_h  = y - 20
            end_h = y + h + 20
            
            if end_h > img_h: end_h = y + h
            if start_h < 0: start_h = y

            res = vis[start_h:end_h, start_w:end_w]
            return res, (x, y, w, h)


    return None, None
