import cv2 as cv
import numpy as np
import math
import json
from operator import attrgetter

class GarisAtas:
    def __init__(self):
        self.basic = ''

    def basicInfo(self, image):
        # image = cv.imread("src/dion2.jpeg")

        if image.shape[1] > 700:
            scale_percent = 18 # percent of original size
        else:
            scale_percent = 100

        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        image = cv.resize(image, dim, interpolation = cv.INTER_AREA)

        kernel = np.ones((3,3), np.uint8)

        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        lowerskin = np.array([0, 20, 70], dtype=np.uint8)
        upperskin = np.array([20, 255, 255], dtype=np.uint8)

        mask = cv.inRange(hsv, lowerskin, upperskin)
        mask = cv.dilate(mask, kernel, iterations = 4)
        mask = cv.GaussianBlur(mask, (5,5), 100)

        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        cnt = max(contours, key = lambda x: cv.contourArea(x))

        epsilon = 0.0005*cv.arcLength(cnt,True)
        approx = cv.approxPolyDP(cnt, epsilon, True)

        hull = cv.convexHull(cnt)

        areahull = cv.contourArea(hull)
        areacnt = cv.contourArea(cnt)
        cv.drawContours(mask, [cnt], 0, (0, 255, 0), 2)

        cv.drawContours(mask, [hull], 0, (0, 255, 0), 2)
        hull = cv.convexHull(approx, returnPoints=False)
        defects = cv.convexityDefects(approx, hull)

        l=0

        titikForPrint = []
        titikForProccess = []

        for i in range(defects.shape[0]):
            s,e,f,d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100, 100)

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))

            d = (2*ar)/a

            angle = math.acos((b**2 + c**2 - a**2) / (2*b*c)) * 57

            if angle <=90 and d>30:
                l += 1
                cv.circle(mask, far, 3, [255,0,0], -1)
                titikForPrint.append({
                    "x": int(far[0]),
                    "y": int(far[1])
                })
                titikForProccess.append(Coord(int(far[0]), int(far[1])))
            
            cv.line(mask, start, end, [255, 255, 255], 2)
        
        l += 1

        # cv.imshow('mask', mask)
        # cv.waitKey(0)
        tangan = ''
        pangkal = ''
        if len(titikForProccess) != 4:
            pangkal = 'cannot detect hand'
            tangan = 'cannot detect hand'
        else:
            maximumy = max(titikForProccess, key=attrgetter('y'))
            minimumx = min(titikForProccess, key=attrgetter('x'))
            # print(maximumy.y)
            # print(minimumx.y)
            if minimumx.x == maximumy.x:
                tangan = 'kiri'
            else:
                tangan = 'kanan'

            titikForProccess = [s for s in titikForProccess if s.y != maximumy.y]
            sortedTitik = sorted(titikForProccess, key=lambda var: var.x)
            titikY = []
            for var in sortedTitik:
                titikY.append(int(var.y))

            if (titikY[0] == titikY[1]) and (titikY[1] == titikY[2]):
                pangkal = 'lurus'
            else:
                expect = (titikY[0] + titikY[2]) / 2
                realError1 = titikY[1] - 3
                realError2 = titikY[1] + 3
                if (expect >= float(realError1)) and (expect <= float(realError2)):
                    pangkal = 'lurus'
                else:
                    pangkal = 'tidak lurus'

        return {
            'titik': titikForPrint,
            'tangan': tangan,
            'pangkalJari': pangkal
        }

class Coord:
    def __init__(self, x, y):
        self.x = x
        self.y = y