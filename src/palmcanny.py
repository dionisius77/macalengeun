from flask import Flask
from flask_restful import Resource, Api
import cv2 as cv
import numpy as np
import base64 
import json
import random as rng

class PalmCanny(Resource):
    def get(self):
        result = []
        image = cv.imread("src/tangan_1.jpg")
        print(image.shape[1])
        if image.shape[1] < 1500:
            front = 90
            back = 90
        else:
            front = 30
            back = 30
        # black & white processing
        blur = cv.GaussianBlur(image, (3,3), 0)
        gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        # canny processing
        edges = cv.Canny(gray,front,back,apertureSize = 3)

        kernel = np.ones((3,3),np.uint8)
        # original = cv.dilate(gray, kernel, iterations = 10)
        # original = cv.erode(original, kernel, iterations = 10)
        original = cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)

        if image.shape[1] > 2000:
            scale_percent = 20 # percent of original size
        else:
            scale_percent = 100
            
        width = int(edges.shape[1] * scale_percent / 100)
        height = int(edges.shape[0] * scale_percent / 100)
        dim = (width, height)
        # invert negative
        resized = cv.bitwise_not(edges)

        # Find contours
        ret, thresh = cv.threshold(resized, 150, 300, cv.THRESH_BINARY_INV)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Draw contours
        drawing = np.zeros((resized.shape[0], resized.shape[1], 3), dtype=np.uint8)
        # drawing2 = np.zeros((resized.shape[0], resized.shape[1], 3), dtype=np.uint8)
        # for i in range(len(contours)):
        #     color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        #     cv.drawContours(drawing, contours, i, color, 5, cv.LINE_8, hierarchy, 2)

        # for cnt in contours:
        #     hull = cv.convexHull(cnt)
        #     cv.drawContours(drawing, [hull], -1, (255, 255, 255), 1)

        cv.drawContours(drawing, contours, -1, 255, 1)
            
        # resizedDrawing2 = cv.resize(drawing2, dim, interpolation = cv.INTER_AREA)
        resizedDrawing = cv.resize(drawing, dim, interpolation = cv.INTER_AREA)
        resizedImage = cv.resize(original, dim, interpolation = cv.INTER_AREA)

        cv.imshow("Palm line", np.hstack([resizedDrawing, resizedImage])) # to generate view the palm
        cv.waitKey(0)
        return {"result" : result}