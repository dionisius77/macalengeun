from flask import Flask
from flask_restful import Resource, Api
import cv2 as cv
import numpy as np
import base64 
import json
import random as rng

class MorphCanny(Resource):
    def get(self, threshold=25):
        
        image = cv.imread("src/tangan_1.jpg",1)

        if image.shape[1] > 2000:
            scale_percent = 18 # percent of original size
        else:
            scale_percent = 100

        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        image = cv.resize(image, dim, interpolation = cv.INTER_AREA)

        color = (0, 0, 0)
        thickness = 2

        left, right, bottom, top, start_point, end_point, thresh, blur = RectProcessing.getRect(image)

        croped = cv.rectangle(image, start_point, end_point, color, thickness)

        cv.circle(image, left, 8, (0, 50, 255), -1)
        cv.circle(image, right, 8, (0, 255, 255), -1)
        cv.circle(image, top, 8, (255, 50, 0), -1)
        cv.circle(image, bottom, 8, (255, 255, 0), -1)

        final = cv.bitwise_and(image, image, mask=thresh)

        cv.imshow('image', blur)
        cv.imshow('thresh', final)
        cv.waitKey(0)

        # print('kiri: {}'.format(left))
        # print('kanan: {}'.format(right))
        # print('atas: {}'.format(top))
        # print('bawah: {}'.format(bottom))

class RectProcessing:
    def getRect(image):
        cotrast_gray = cv.addWeighted(image, 1.0, np.zeros(image.shape, image.dtype), 0, 0)
        blur = cv.GaussianBlur(cotrast_gray, (5,5), 0)
        gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)


        thresh = cv.threshold(gray, 150, 200, cv.THRESH_BINARY_INV)[1]

        cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        c = max(cnts, key=cv.contourArea)

        left = tuple(c[c[:, :, 0].argmin()][0])
        right = tuple(c[c[:, :, 0].argmax()][0])
        top = tuple(c[c[:, :, 1].argmin()][0])
        bottom = tuple(c[c[:, :, 1].argmax()][0])

        # croped = image[right+top, left+bottom]
        start_point = (left[0], top[1])
        end_point = (right[0], bottom[1])
        
        final = cv.drawContours(image, [c], -1, (0, 255, 0), 3)

        return (left, right, bottom, top, start_point, end_point, thresh, blur)