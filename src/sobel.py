from flask import Flask
from flask_restful import Resource, Api
import cv2 as cv
import numpy as np
from src.imageProcessing import ImageProcessing

class Sobel(Resource):
    def get(self):
        image = cv.imread("src/tangan_1.jpg")

        cannyProcess = ImageProcessing.cannyProcess(image)
        contourProcess = ImageProcessing.contourProcess(cannyProcess)
        dilated = ImageProcessing.dilate(contourProcess)

        resizedContour = ImageProcessing.resizeImage(contourProcess)
        resizedCanny = ImageProcessing.resizeImage(cannyProcess)
        resizedDilate = ImageProcessing.resizeImage(dilated)

        # blur = cv.resize(blur, dim, interpolation = cv.INTER_AREA)

        # status = cv.imwrite('C:/Users/c22440/Documents/Barrans/project/palmline/src/contour.jpeg', resizedContour)
        # print(status)
        cv.imshow("Contour", resizedContour) # to generate view the palm
        cv.imshow("Canny", resizedCanny)
        cv.imshow("Dilated", resizedDilate)
        cv.waitKey(0)
        return {}