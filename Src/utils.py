import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
import sys
import cv2 as cv2
import pickle
import os
import matplotlib.pyplot as plt

def drawRects(orgImg, rects, GTrects=None, numShowRects=100):
	# orgImg -> openCV file
	imOut = orgImg.copy()

	# itereate over all the region proposals
	for i in range(numShowRects):
		# draw rectangle for region proposal till numShowRects
		x, y, w, h = rects[i]
		cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

	if GTrects:
		for rect in GTrects:
			x, y, w, h = rect[:4]
			cv2.rectangle(imOut, (x, y), (x + w, y + h), (255, 0, 0), 1, cv2.LINE_AA)

	# show output
	plt.figure()
	plt.imshow(cv2.cvtColor(imOut, cv2.COLOR_BGR2RGB))
	plt.show()

def quick_imshow(img):
	plt.figure()
	plt.imshow(img)
	plt.show()
