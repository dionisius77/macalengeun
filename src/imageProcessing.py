import cv2 as cv
import numpy as np

class ImageProcessing():
    def resizeImage(image):
        # resize
        if image.shape[1] > 1500:
            scale_percent = 18 # percent of original size
        else:
            scale_percent = 100
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
        return resized

    def cannyProcess(image):
        if image.shape[1] < 1500:
            front = 40
            back = 50
        else:
            front = 10
            back = 15

        blur = cv.blur(image, (6,6))
        gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        cannyProcess = cv.Canny(gray, front, back, apertureSize = 3)

        negatived = ImageProcessing.negative(cannyProcess)

        return negatived

    def negative(image):
        return cv.bitwise_not(image)

    def contourProcess(cannyProcess):
        ret, thresh = cv.threshold(cannyProcess, 150, 300, cv.THRESH_BINARY_INV)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Draw contours
        drawing2 = np.zeros((cannyProcess.shape[0], cannyProcess.shape[1], 3), dtype=np.uint8)
        cv.drawContours(drawing2, contours, -1, (255, 255, 255), 1)

        cnt = sorted(contours, key=cv.contourArea)
        # print(contours[1])
        # for cnt in contours:
        print(len(cnt))
        # print(cnt[2])
        # for i in range(200, 300):
        #     hull = cv.convexHull(contours[i])
        #     cv.drawContours(drawing2, cnt[i], -4, (255, 255, 255), 2)
        return drawing2

    def dilate(image):
        kernel = np.ones((1,1), np.uint8)
        dilated = cv.dilate(image, kernel, iterations = 8)
        return dilated