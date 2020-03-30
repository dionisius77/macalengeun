from flask import Flask
from flask_restful import Resource, Api
from src.TOOLS import Functions
from src.garis_atas import GarisAtas
import cv2
import numpy as np
import math
import argparse
import base64

class Plate(Resource):
    def get(self):

        img = cv2.imread('src/dion2.jpeg')
        basicInfo = GarisAtas.basicInfo(self, img)

        if img.shape[1] > 700:
            scale_percent = 18 # percent of original size
        else:
            scale_percent = 100

        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        
        # cv2.imshow('gray', value)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
        blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
        # cv2.imshow('topHat', topHat)
        # cv2.imshow('blackHat', blackHat)

        add = cv2.add(value, topHat)
        subtract = cv2.subtract(add, blackHat)
        # cv2.imshow('subtract', subtract)

        blur = cv2.GaussianBlur(subtract, (3, 3), 0)
        # cv2.imshow('blur', blur)

        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
        cv2.imshow('thresh', thresh)

        cv2MajorVersion = cv2.__version__.split(".")[0]
        # check for contours on thresh
        # print(cv2MajorVersion)
        if int(cv2MajorVersion) >= 4:
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        height, width = thresh.shape

        imageContours = np.zeros((height, width, 3), dtype=np.uint8)

        possibleChars = []
        possibleCharsData = []
        countOfPossibleChars = 0

        for i in range(0, len(contours)):

            cv2.drawContours(imageContours, contours, i, (255, 255, 255))

            possibleChar = Functions.ifChar(contours[i])

            if Functions.checkIfChar(possibleChar) is True:
                dataTest = Functions.checkIfCharData(possibleChar)
                countOfPossibleChars = countOfPossibleChars + 1
                possibleChars.append(possibleChar)
                possibleCharsData.append(dataTest)

        # cv2.imshow("contours", imageContours)

        imageContours = np.zeros((height, width, 3), np.uint8)

        ctrs = []

        for char in possibleChars:
            ctrs.append(char.contour)

        # using values from ctrs to draw new contours
        # cv2.drawContours(imageContours, ctrs, -4, (255, 255, 255))
        cv2.fillPoly(imageContours, pts = ctrs, color= (255, 255, 255))
        cv2.imshow("contoursPossibleChars", imageContours)

        plates_list = []
        listOfListsOfMatchingChars = []

        for possibleC in possibleChars:
            def matchingChars(possibleC, possibleChars):
                listOfMatchingChars = []

                for possibleMatchingChar in possibleChars:
                    if possibleMatchingChar == possibleC:
                        continue

                    distanceBetweenChars = Functions.distanceBetweenChars(possibleC, possibleMatchingChar)

                    angleBetweenChars = Functions.angleBetweenChars(possibleC, possibleMatchingChar)

                    changeInArea = float(abs(possibleMatchingChar.boundingRectArea - possibleC.boundingRectArea)) / float(
                        possibleC.boundingRectArea)

                    changeInWidth = float(abs(possibleMatchingChar.boundingRectWidth - possibleC.boundingRectWidth)) / float(
                        possibleC.boundingRectWidth)

                    changeInHeight = float(abs(possibleMatchingChar.boundingRectHeight - possibleC.boundingRectHeight)) / float(
                        possibleC.boundingRectHeight)

                    if distanceBetweenChars < (possibleC.diagonalSize * 30) and \
                            (angleBetweenChars < 40.0 and angleBetweenChars > 10.0) and \
                            (changeInArea < 40.0 and changeInArea > 1.0) and \
                            changeInWidth < 40.0 and \
                            (changeInHeight < 40.0 and changeInHeight > 1.4):
                    # if distanceBetweenChars < (possibleC.diagonalSize * 5) and \
                    #         angleBetweenChars < 12.0 and \
                    #         changeInArea < 0.5 and \
                    #         changeInWidth < 0.8 and \
                    #         changeInHeight < 0.2:
                        listOfMatchingChars.append(possibleMatchingChar)
                        # print(changeInWidth)
                        # print("=================")

                return listOfMatchingChars

            listOfMatchingChars = matchingChars(possibleC, possibleChars)

            if len(listOfMatchingChars) < 3:
                continue

            listOfListsOfMatchingChars.append(listOfMatchingChars)

            listOfPossibleCharsWithCurrentMatchesRemoved = list(set(possibleChars) - set(listOfMatchingChars))

            recursiveListOfListsOfMatchingChars = []

            for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
                listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)

            break

        for listOfMatchingChars in listOfListsOfMatchingChars:
            possiblePlate = Functions.PossiblePlate()

            listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.centerX)

            plateCenterX = (listOfMatchingChars[0].centerX + listOfMatchingChars[len(listOfMatchingChars) - 1].centerX) / 2.0
            plateCenterY = (listOfMatchingChars[0].centerY + listOfMatchingChars[len(listOfMatchingChars) - 1].centerY) / 2.0

            plateCenter = plateCenterX, plateCenterY

            plateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].boundingRectX + listOfMatchingChars[
                len(listOfMatchingChars) - 1].boundingRectWidth - listOfMatchingChars[0].boundingRectX) * 1.3)

            totalOfCharHeights = 0

            for matchingChar in listOfMatchingChars:
                totalOfCharHeights = totalOfCharHeights + matchingChar.boundingRectHeight

            averageCharHeight = totalOfCharHeights / len(listOfMatchingChars)

            plateHeight = int(averageCharHeight * 1.5)

            opposite = listOfMatchingChars[len(listOfMatchingChars) - 1].centerY - listOfMatchingChars[0].centerY

            hypotenuse = Functions.distanceBetweenChars(listOfMatchingChars[0],
                                                        listOfMatchingChars[len(listOfMatchingChars) - 1])
            correctionAngleInRad = math.asin(opposite / hypotenuse)
            correctionAngleInDeg = correctionAngleInRad * (180.0 / math.pi)

            possiblePlate.rrLocationOfPlateInScene = (tuple(plateCenter), (plateWidth, plateHeight), correctionAngleInDeg)

            rotationMatrix = cv2.getRotationMatrix2D(tuple(plateCenter), correctionAngleInDeg, 1.0)

            height, width, numChannels = img.shape

            imgRotated = cv2.warpAffine(img, rotationMatrix, (width, height))

            imgCropped = cv2.getRectSubPix(imgRotated, (plateWidth, plateHeight), tuple(plateCenter))

            possiblePlate.Plate = imgCropped

            if possiblePlate.Plate is not None:
                plates_list.append(possiblePlate)

        cv2.waitKey(0)
        _, buffer = cv2.imencode('.jpg', imageContours)
        bitImage = base64.urlsafe_b64encode(buffer)
        base64Image = str(bitImage, "utf-8")

        return {
            'basicInfo': basicInfo,
            'possibleLine': possibleCharsData,
            'imageData': base64Image
        }