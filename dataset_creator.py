#!/usr/bin/env python3


import numpy as np
import cv2
import os
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import pickle
from glob import glob
import imgaug as ia
from imgaug import augmenters as iaa
from shapely.geometry import Polygon
import Augmentor

"""

Backgrounds file:
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar xf dtd-r1.0.1.tar.gz to dtd folder.


"""

cv2_resource_path = "./venv/lib/python3.7/site-packages/cv2/data/"

data_dir = "data"  # Directory that will contain all kinds of data (the data we download and the data we generate)

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

card_suits = ['s', 'h', 'd', 'c']
card_values = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']

# Pickle file containing the background images from the DTD
backgrounds_pck_fn = data_dir + "/backgrounds.pck"

# Pickle file containing the card images
cards_pck_fn = data_dir + "/cards.pck"

# imgW,imgH: dimensions of the generated dataset images
imgW = 720
imgH = 720

# nealcards
""" NB. the corners on our card set is not consistent. so I will choose the most inclusive area.
further the measurements asked seem wrong Ymax should be inclusive of Ymin."""
cardW = 56
cardH = 86
cornerXmin = 2
cornerXmax = 8
cornerYmin = 3
cornerYmax = 21

xml_body_1 = """<annotation>
        <folder>FOLDER</folder>
        <filename>{FILENAME}</filename>
        <path>{PATH}</path>
        <source>
                <database>Unknown</database>
        </source>
        <size>
                <width>{WIDTH}</width>
                <height>{HEIGHT}</height>
                <depth>3</depth>
        </size>
"""
xml_object = """ <object>
                <name>{CLASS}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>{XMIN}</xmin>
                        <ymin>{YMIN}</ymin>
                        <xmax>{XMAX}</xmax>
                        <ymax>{YMAX}</ymax>
                </bndbox>
        </object>
"""
xml_body_2 = """</annotation>        
"""

"""
Various test images paths:

    #img=cv2.imread("./test/all_mads.jpg")
    #img=cv2.imread("./test/depositphotos_184840322-stock-photo-single-spades-playing-card-gamble.jpg")
    #img=cv2.imread("./test/96066795-single-of-spades-playing-card-for-gamble-playing-cards-2-isolated-on-black-background-great-for-any-.jpg")
    #img=cv2.imread("./test/CW_Cards_Africanqueen.jpg")
    squashpath = "./test/squash2.jpg"
    squash = cv2.imread(squashpath)

"""


def picture():
    img = cv2.imread("./test/green_screen.png")
    cv2.imshow("output", img)
    cv2.waitKey(0)


def video():
    cap = cv2.VideoCapture("test/2c.avi")
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def imgconversiongray():
    img = cv2.imread("./test/green_screen.png")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    cv2.imshow("output", imgGray)
    cv2.waitKey(0)


def imgconversiongauss():
    img = cv2.imread("./test/green_screen.png")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
    cv2.imshow("output", imgBlur)
    cv2.waitKey(0)


def imgconversionCanny():
    img = cv2.imread("./test/green_screen.png")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
    imgCanny = cv2.Canny(img, 100, 100)
    cv2.imshow("output", imgCanny)
    cv2.waitKey(0)
    imgCanny2 = cv2.Canny(imgGray, 100, 100)
    cv2.imshow("output", imgCanny2)
    cv2.waitKey(0)
    imgCanny3 = cv2.Canny(imgBlur, 100, 200)
    cv2.imshow("output", imgCanny3)
    cv2.waitKey(0)


def imgconversionDialation():
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.imread("./test/green_screen.png")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
    imgCanny = cv2.Canny(img, 100, 100)
    imgDialation = cv2.dilate(imgCanny, kernel, iterations=5)
    cv2.imshow("output", imgDialation)
    cv2.waitKey(0)


def imgErosion():
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.imread("./test/green_screen.png")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
    imgCanny = cv2.Canny(img, 100, 100)
    imgDialation = cv2.dilate(imgCanny, kernel, iterations=5)
    cv2.imshow("output", imgDialation)
    cv2.waitKey(0)
    imgErosi = cv2.dilate(imgDialation, kernel, iterations=5)
    cv2.imshow("output", imgErosi)
    cv2.waitKey(0)


def imageResize():
    img = cv2.imread("./test/green_screen.png")
    print(img.shape)
    imgResize = cv2.resize(img, (300, 200))
    print(imgResize.shape)
    cv2.imshow("output", imgResize)
    cv2.waitKey(0)


def imageCrop():
    img = cv2.imread("./test/green_screen.png")
    print(img.shape)
    # - y, then - x.
    imgCrop = img[100:200, 1000:1200]
    cv2.imshow("output", imgCrop)
    cv2.waitKey(0)


def imageTrim(img, top=0, bottom=0, left=0, right=0):
    # - y, then - x.
    ylength = img.shape[0]
    xlength = img.shape[1]
    imgCrop = img[0 + top:ylength - bottom, 0 + left: xlength - right]
    imgCrop = cv2.resize(imgCrop, (xlength, ylength))
    return imgCrop


def createImg():
    # np dimensions, then type = np.uint8 = 0-255
    img = np.zeros((512, 512, 3), np.uint8)
    print(img.shape)

    img[100:200, 200:250] = 255, 0, 0
    img[200:300, 200:250] = 0, 255, 0
    img[300:400, 200:250] = 0, 0, 255

    # img x,y start, x,y finish
    # cv2.line(img, (0, 0), (img.shape[1], 500), (0, 0, 255), (5))

    cv2.rectangle(img, (1, 1), (250, 350), (0, 0, 255), 2, cv2.FILLED)

    cv2.circle(img, (250, 350), 30, (0, 0, 255))
    cv2.putText(img, " PUFFIN ", (300, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 1)

    cv2.imshow("output", img)
    cv2.waitKey(0)


def warpImage():
    img = cv2.imread("./test/squash2.jpg")
    img2 = cv2.imread("./test/green_screen.png")
    print(img.shape)

    cornerA = [1350, 1720]
    cornerB = [2290, 1850]
    cornerC = [440, 2580]
    cornerD = [1885, 2980]
    cv2.circle(img, (1350, 1720), 20, (255, 0, 255), 5)
    cv2.circle(img, (2290, 1850), 20, (0, 0, 255), 5)
    cv2.circle(img, (440, 2580), 20, (0, 255, 255), 5)
    cv2.circle(img, (1885, 2980), 20, (255, 0, 0), 5)
    pts1 = (np.float32([cornerA, cornerB, cornerC, cornerD]))
    # new size
    width = 57 * 10
    height = 87 * 10
    pts2 = (np.float32([[0, 0], [width, 0], [0, height], [width, height]]))
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))
    cv2.imshow("output2", img2)
    cv2.imshow("outputpic", img)
    cv2.imshow("output", imgOutput)
    cv2.waitKey(0)


def stackimage(img):
    imgHor = np.hstack((img, img))
    imgVer = np.vstack((img, img))
    cv2.imshow(imgHor)
    cv2.imshow(imgVer)
    cv2.waitKey(0)


def empty(arg):
    pass


def imageHSV():
    path = "./test/squash2.jpg"
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 1200, 240)
    cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
    cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

    while True:
        img = cv2.imread(path)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
        print(h_min, h_max, s_min, s_max, v_min, v_max)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        # find out what this does - mask
        mask = cv2.inRange(imgHSV, lower, upper)

        cv2.imshow("Original", img)
        cv2.imshow("HSV", imgHSV)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1000)

    cv2.destroyAllWindows()


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def shape_recognition():
    pathsq = "./test/squash2.jpg"
    imgsq = cv2.imread(pathsq)
    img = cv2.imread("./test/all_mads.jpg")
    img = cv2.imread("./dataset2_blackbackground/2c.jpg")
    # img=cv2.imread("./data/cards/2c-min.jpg")
    # img=cv2.imread("./data/cards/sshot2.png")
    # img=cv2.imread("/Users/n/PycharmProjects/playing-card-detection/data/cards/2c-sizemin.jpg")
    # img=cv2.imread("./test/depositphotos_184840322-stock-photo-single-spades-playing-card-gamble.jpg")
    # img=cv2.imread("./test/96066795-single-of-spades-playing-card-for-gamble-playing-cards-2-isolated-on-black-background-great-for-any-.jpg")
    # img=cv2.imread("./test/CW_Cards_Africanqueen.jpg")
    # img = cv2.imread("./data/cards/2c-min.jpg")

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    # imgCanny = cv2.Canny(imgGray, 0, 197)
    imgCanny = cv2.Canny(imgBlur, 0, 197)
    imgBlank = np.zeros_like(img)
    imgContours = img.copy()

    getContours(imgCanny, imgContours, imgBlank)

    imgsqGray = cv2.cvtColor(imgsq, cv2.COLOR_BGRA2GRAY)
    imgsqBlur = cv2.GaussianBlur(imgsqGray, (7, 7), 1)
    imgsqCanny = cv2.Canny(imgsqBlur, 20, 10)
    imgsqBlank = np.zeros_like(imgsq)
    imgsqContours = imgsq.copy()

    # getContours(imgsqCanny, imgsqContours,imgsqBlank)

    imgStack = stackImages(.7, [img, imgBlur, imgCanny, imgContours, imgBlank])
    # imgStack = stackImages(.7, [img, imgCanny, imgBlank])
    # imgsqStack = stackImages(0.2, [imgsq, imgsqBlur, imgsqCanny, imgsqContours, imgsqBlank])

    cv2.imshow("Stack", imgStack)
    # cv2.imshow("Stacksq", imgsqStack)
    cv2.waitKey(0)


def getContours(img, imgContour, imgBlank):
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 10)
            peri = cv2.arcLength(cnt, True)
            # print(peri)
            # looking at connected shapes - this is the TRUE
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # len here gives how many connected edges.
            objCor = (len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            objectType = ""
            cardAspectRatio = 57 / 87
            devation = 0.10
            if objCor == 4:
                aspectRatio = w / float(h)
                if aspectRatio > (1 - devation) * cardAspectRatio and aspectRatio < (1 + devation) * cardAspectRatio:
                    objectType = "Card"
            cv2.rectangle(imgBlank, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgBlank, objectType,
                        (x + (w // 2) - 20, y + (h // 2)), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)


# NB might need to create custom cascades.


def face_recognition():
    """NB. file has been removed."""
    face_cascade = cv2.CascadeClassifier(cv2_resource_path + "haarcascade_frontalface_default.xml")
    img = cv2.imread("IMG_4703.JPG")

    img_gray = cv2.cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Result", img)
    cv2.waitKey(0)


def getContours_dataset2(img, imgContour):
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 10)
            peri = cv2.arcLength(cnt, True)
            # print(peri)
            # looking at connected shapes - this is the TRUE
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # len here gives how many connected edges.
            objCor = (len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            objectType = ""
            cardAspectRatio = 57 / 87
            devation = 0.10
            if objCor == 4:
                aspectRatio = w / float(h)
                if aspectRatio > (1 - devation) * cardAspectRatio and aspectRatio < (1 + devation) * cardAspectRatio:
                    objectType = "Card"
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, objectType,
                        (x + (w // 2) - 20, y + (h // 2)), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)


def card_prep():
    img1 = cv2.imread("./dataset2_blackbackground/4d.jpg")
    img2 = cv2.imread("./dataset2_blackbackground/As.jpg")
    img3 = cv2.imread("./dataset2_blackbackground/Ac.jpg")
    imgStack = stackImages(0.4, [img1, img2, img3])

    def callback(foo):
        pass

    # create windows and trackbar
    cv2.namedWindow('parameters')
    cv2.createTrackbar('threshold1', 'parameters', 0, 255, callback)  # change the maximum to whatever you like
    cv2.createTrackbar('threshold2', 'parameters', 0, 255, callback)  # change the maximum to whatever you like
    cv2.createTrackbar('apertureSize', 'parameters', 0, 2, callback)
    cv2.createTrackbar('L1/L2', 'parameters', 0, 1, callback)

    while (True):
        # get threshold value from trackbar
        th1 = cv2.getTrackbarPos('threshold1', 'parameters')
        th2 = cv2.getTrackbarPos('threshold2', 'parameters')

        # aperture size can only be 3,5, or 7
        apSize = cv2.getTrackbarPos('apertureSize', 'parameters') * 2 + 3

        # true or false for the norm flag
        norm_flag = cv2.getTrackbarPos('L1/L2', 'parameters') == 1

        # print out the values
        print('')
        print('threshold1: {}'.format(th1))
        print('threshold2: {}'.format(th2))
        print('apertureSize: {}'.format(apSize))
        print('L2gradient: {}'.format(norm_flag))

        cv2.imshow("orginal", imgStack)
        gray = cv2.cvtColor(imgStack, cv2.COLOR_BGRA2GRAY)
        # Convert in gray color
        # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Noise-reducing and edge-preserving filter
        gray = cv2.bilateralFilter(gray, 5, 75, 75)
        # Edge extraction
        # edge=cv2.Canny(gray, 0, 197, apertureSize=3)
        imgGray = cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
        imgCanny = cv2.Canny(imgGray, 0, 197)
        imgCanny = cv2.Canny(gray, th1, th2, apertureSize=apSize, L2gradient=norm_flag)
        # imgCanny = cv2.Canny(imgBlur, 0, 197)
        imgBlank = np.zeros_like(img1)
        imgContours = img1.copy()

        getContours(imgCanny, imgContours, imgBlank)
        # getContours_dataset2(gray, edge)
        imgStack = stackImages(.4, [img1, imgBlur, imgCanny, imgContours, imgBlank])
        # edge = cv2.Canny(gray, th1, th2, apertureSize=apSize, L2gradient=norm_flag)
        cv2.imshow('stack', imgStack)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def card_prep2(imgpath):
    # img1 = cv2.imread("./dataset2_blackbackground/4d.jpg")
    # img1 = cv2.imread("./dataset2_blackbackground/As.jpg")
    # img1 = cv2.imread("./dataset2_blackbackground/2c.jpg")
    img1 = cv2.imread(imgpath)

    ylength = int(img1.shape[0] / 8)
    xlength = int(img1.shape[1] / 8)
    img1 = cv2.resize(img1, (xlength, ylength))

    # imgStack = stackImages(0.4, [img1, img2, img3])
    # cv2.imshow("orginal", imgStack)

    def callback(foo):
        pass

    # create windows and trackbar
    cv2.namedWindow('parameters')
    cv2.createTrackbar('threshold1', 'parameters', 0, 255, callback)  # change the maximum to whatever you like
    cv2.createTrackbar('threshold2', 'parameters', 0, 255, callback)  # change the maximum to whatever you like
    cv2.createTrackbar('apertureSize', 'parameters', 0, 2, callback)
    cv2.createTrackbar('L1/L2', 'parameters', 0, 1, callback)
    while (True):
        # get threshold value from trackbar
        th1 = cv2.getTrackbarPos('threshold1', 'parameters')
        th2 = cv2.getTrackbarPos('threshold2', 'parameters')

        # aperture size can only be 3,5, or 7
        apSize = cv2.getTrackbarPos('apertureSize', 'parameters') * 2 + 3

        # true or false for the norm flag
        norm_flag = cv2.getTrackbarPos('L1/L2', 'parameters') == 1

        # print out the values
        print('')
        print('threshold1: {}'.format(th1))
        print('threshold2: {}'.format(th2))
        print('apertureSize: {}'.format(apSize))
        print('L2gradient: {}'.format(norm_flag))

        # cv2.imshow("orginal", imgStack)
        # gray = cv2.cvtColor(imgStack, cv2.COLOR_BGRA2GRAY)
        # gray = cv2.bilateralFilter(gray, 5, 75, 75)
        imgGray = cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
        # imgCanny = cv2.Canny(imgGray, 0, 197, apertureSize=3)
        imgCanny = cv2.Canny(imgBlur, th1, th2, apertureSize=apSize, L2gradient=norm_flag)
        imgBlank = np.zeros_like(img1)
        imgContours = img1.copy()

        getContours(imgCanny, imgContours, imgBlank)

        # cv2.imshow('Contours', imgContours)
        # cv2.imshow('imgBlank', imgBlank)
        imgStack1 = stackImages(.7, [img1, imgBlur, imgCanny, imgContours, imgBlank])
        cv2.imshow('stack', imgStack1)
        if cv2.waitKey(2000) & 0xFF == ord('q'):
            break


# card_prep2()

def card_extract(img, output_fn=None, crop=0):
    # going to try to count pixels for zoom, - pixels to mm is 24
    # card settings:
    # nealcards
    cardW = 56
    cardH = 86
    cornerXmin = 2
    cornerXmax = 8
    cornerYmin = 4
    cornerYmax = 21

    # We convert the measures from mm to pixels: multiply by an arbitrary factor 'zoom'
    # You shouldn't need to change this
    # zoom = 24
    zoom = 4
    cardW *= zoom
    cardH *= zoom
    cornerXmin = int(cornerXmin * zoom)
    cornerXmax = int(cornerXmax * zoom)
    cornerYmin = int(cornerYmin * zoom)
    cornerYmax = int(cornerYmax * zoom)

    # misc variables from jupyter
    # imgW,imgH: dimensions of the generated dataset images
    imgW = 720
    imgH = 720

    refCard = np.array([[0, 0], [cardW, 0], [cardW, cardH], [0, cardH]], dtype=np.float32)
    refCardRot = np.array([[cardW, 0], [cardW, cardH], [0, cardH], [0, 0]], dtype=np.float32)
    refCornerHL = np.array(
        [[cornerXmin, cornerYmin], [cornerXmax, cornerYmin], [cornerXmax, cornerYmax], [cornerXmin, cornerYmax]],
        dtype=np.float32)
    refCornerLR = np.array([[cardW - cornerXmax, cardH - cornerYmax], [cardW - cornerXmin, cardH - cornerYmax],
                            [cardW - cornerXmin, cardH - cornerYmin], [cardW - cornerXmax, cardH - cornerYmin]],
                           dtype=np.float32)
    refCorners = np.array([refCornerHL, refCornerLR])

    # alphamask:
    bord_size = 2  # bord_size alpha=0
    alphamask = np.ones((cardH, cardW), dtype=np.uint8) * 255
    cv2.rectangle(alphamask, (0, 0), (cardW - 1, cardH - 1), 0, bord_size)
    cv2.line(alphamask, (bord_size * 3, 0), (0, bord_size * 3), 0, bord_size)
    cv2.line(alphamask, (cardW - bord_size * 3, 0), (cardW, bord_size * 3), 0, bord_size)
    cv2.line(alphamask, (0, cardH - bord_size * 3), (bord_size * 3, cardH), 0, bord_size)
    cv2.line(alphamask, (cardW - bord_size * 3, cardH), (cardW, cardH - bord_size * 3), 0, bord_size)
    plt.figure(figsize=(10, 10))
    plt.imshow(alphamask)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 0, 197, apertureSize=3, L2gradient=True)
    # imgCanny = cv2.Canny(imgBlur, 0, 197, apertureSize=3)
    contours, hierachy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    areaCnt = cv2.contourArea(cnt)
    areaBox = cv2.contourArea(box)

    ((xr, yr), (wr, hr), thetar) = rect

    if wr > hr:
        Mp = cv2.getPerspectiveTransform(np.float32(box), refCard)
    else:
        Mp = cv2.getPerspectiveTransform(np.float32(box), refCardRot)

    imgwarp = cv2.warpPerspective(img, Mp, (cardW, cardH))
    # Add alpha layer
    imgwarp = cv2.cvtColor(imgwarp, cv2.COLOR_BGR2BGRA)
    # Shape of 'cnt' is (n,1,2), type=int with n = number of points
    # We reshape into (1,n,2), type=float32, before feeding to perspectiveTransform
    cnta = cnt.reshape(1, -1, 2).astype(np.float32)
    # Apply the transformation 'Mp' to the contour
    cntwarp = cv2.perspectiveTransform(cnta, Mp)
    cntwarp = cntwarp.astype(np.int)

    # We build the alpha channel so that we have transparency on the
    # external border of the card
    # First, initialize alpha channel fully transparent
    alphachannel = np.zeros(imgwarp.shape[:2], dtype=np.uint8)
    # Then fill in the contour to make opaque this zone of the card
    cv2.drawContours(alphachannel, cntwarp, 0, 255, -1)

    # Apply the alphamask onto the alpha channel to clean it
    alphachannel = cv2.bitwise_and(alphachannel, alphamask)

    # Add the alphachannel to the warped image
    imgwarp[:, :, 3] = alphachannel
    if crop != 0:
        img1 = imageTrim(imgwarp, crop, crop, crop, crop)
        imgwarp = img1
    # Save the image to file
    if output_fn is not None:
        cv2.imwrite(output_fn, imgwarp)

    return imgwarp


# img1 = cv2.imread("./dataset2_blackbackground/2c.jpg")
# img2 = cv2.imread("./dataset2_blackbackground/2h.jpg")
# img3 = cv2.imread("./dataset2_blackbackground/4d.jpg")
# #
# cardexample = card_extract(img3)
# cv2.imshow('Contours', cardexample)
# cv2.waitKey(0)


def extract_all(dir):
    card_suits = ['s', 'h', 'd', 'c']
    card_values = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
    dir = dir
    extension = "jpg"
    imgs_dir = "data/cards"
    for suit in card_suits:
        for value in card_values:

            card_name = value + suit
            print("extracting: " + card_name)
            file = os.path.join(dir, card_name + "." + extension)
            output_dir = os.path.join(imgs_dir, card_name)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            output_file = output_dir + "/" + card_name + ".jpg"
            img = cv2.imread(file)
            card_extract(img, output_file, crop=2)


# img2=cv2.imread("data/cards/10s/10s.jpg")
# cv2.imshow("test",img2)
# cv2.waitKey(0)


def augment_images(number):
    print("aug")
    imgs_dir = "data/cards"
    imgs_fns = glob(imgs_dir + "/*")
    print(imgs_fns)
    for img in imgs_fns:
        print(img)
        augment_function(img, number)


def augment_function(img, number):
    # print(type(img))
    p = Augmentor.Pipeline(img, output_directory='')
    # p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    # p.zoom(probability=1, min_factor=0.8, max_factor=1.5)
    p.random_brightness(probability=0.9, min_factor=0.7, max_factor=1.3)
    p.random_contrast(probability=0.9, min_factor=0.7, max_factor=1.3)
    p.random_color(probability=0.9, min_factor=0.7, max_factor=1.3)
    # p.skew_top_bottom(probability=0.7, magnitude=0.1)
    p.sample(number)
    # print(type(img,))


# augment_images()

# find hulls:
# this function is taken from someone else
# adding some variables to from the jupyter notebook to fix function

"""
This is a test function to find the value part of the card
looks like the the corner it is finding is way off.
will have to code my own solution
will start with upper left.

further it looks like the corner explanation image in the original is wrong.

"""


def findHull_imageAnalysis(img, corner):
    kernel = np.ones((3, 3), np.uint8)
    corner = corner.astype(np.int)
    print(img.shape)

    y = img.shape[0]
    x = img.shape[1]

    cardW = 56
    cardH = 86

    factor_y = int(y / cardH)
    factor_x = int(x / cardW)

    cornerXmin = 3
    cornerXmax = 9
    cornerYmin = 4
    cornerYmax = 21

    # coordinates
    x1 = int(factor_x * cornerXmin)
    x2 = int(factor_x * cornerXmax)
    y1 = int(factor_y * cornerYmin)
    y2 = int(factor_y * cornerYmax)

    cornerA = [x1, y1]
    cornerB = [x1, y2]
    cornerC = [x2, y1]
    cornerD = [x2, y2]

    # top left corner.
    """ NB. the corners on our card set is not consistent. so I will choose the most inclusive area."""
    print("x1 " + str(x1))
    print("x2 " + str(x2))
    print("y2 " + str(y1))
    print("y1 " + str(y2))

    print(cornerA)
    print(cornerB)
    print(cornerC)
    print(cornerD)

    # We will focus on the zone of 'img' delimited by 'corner'
    # x1 = int(corner[0][0])
    # y1 = int(corner[0][1])
    # x2 = int(corner[2][0])
    # y2 = int(corner[2][1])
    # print("x1 " + str(x1))
    # print("x2 " + str(x2))
    # print("y1 " + str(y1))
    # print("y1 " + str(y2))
    w = x2 - x1
    h = y2 - y1
    zone = img[y1:y2, x1:x2]
    print(zone.shape)

    return zone


# imghull = cv2.imread("./data/cards/Kh/Kh.jpg")
# print(type(imghull))
# # cv2.imshow("preimg",imghull)
# # cv2.waitKey(0)
# #cv2.imshow("I", findHull_imageAnalysis(imghull, refCornerLR))
# cv2.imwrite("./test/croptest.jpg",findHull_imageAnalysis(imghull, refCornerLR))
#


refCard = np.array([[0, 0], [cardW, 0], [cardW, cardH], [0, cardH]], dtype=np.float32)
refCardRot = np.array([[cardW, 0], [cardW, cardH], [0, cardH], [0, 0]], dtype=np.float32)

refCornerHL = np.array([[cornerXmin, cornerYmin],
                        [cornerXmax, cornerYmin],
                        [cornerXmax, cornerYmax],
                        [cornerXmin, cornerYmax]], dtype=np.float32)

refCornerLR = np.array([[cardW - cornerXmax, cardH - cornerYmax],
                        [cardW - cornerXmin, cardH - cornerYmax],
                        [cardW - cornerXmin, cardH - cornerYmin],
                        [cardW - cornerXmax, cardH - cornerYmin]], dtype=np.float32)

refCorners = np.array([refCornerHL, refCornerLR])


def findHull(img, corner=refCornerHL, debug="no", test=False):
    """
        this function is taken from Jupyternotebook.
        Find in the zone 'corner' of image 'img' and return, the convex hull delimiting
        the value and suit symbols
        'corner' (shape (4,2)) is an array of 4 points delimiting a rectangular zone,
        takes one of the 2 possible values : refCornerHL or refCornerLR
    """

    """
    edit here: adding in a function that takes the imgsize into account to convert mm into pixels
    """

    y = img.shape[0]
    x = img.shape[1]

    cardW = 56
    cardH = 86

    factor_y = int(y / cardH)
    factor_x = int(x / cardW)
    zoom = (factor_x + factor_y) / 2

    kernel = np.ones((3, 3), np.uint8)
    corner = corner.astype(np.int)

    # Here will will fix the zoom.
    corner[0][0] = corner[0][0] * factor_x
    corner[0][1] = corner[0][1] * factor_y
    corner[2][0] = corner[2][0] * factor_x
    corner[2][1] = corner[2][1] * factor_y

    # We will focus on the zone of 'img' delimited by 'corner'
    x1 = int(corner[0][0])
    y1 = int(corner[0][1])
    x2 = int(corner[2][0])
    y2 = int(corner[2][1])
    w = x2 - x1
    h = y2 - y1
    zone = img[y1:y2, x1:x2].copy()

    strange_cnt = np.zeros_like(zone)
    # edit this to the the new lighting.
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    thld = cv2.Canny(gray, 30, 200)
    thld = cv2.dilate(thld, kernel, iterations=1)

    # cv2.imshow("handled", thld)
    # cv2.waitKey(0)

    # Find the contours
    contours, _ = cv2.findContours(thld.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 30  # We will reject contours with small area. TWEAK, 'zoom' dependant
    min_solidity = 0.3  # Reject contours with a low solidity. TWEAK

    concat_contour = None  # We will aggregate in 'concat_contour' the contours that we want to keep

    ok = True
    for c in contours:
        area = cv2.contourArea(c)

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        # Determine the center of gravity (cx,cy) of the contour
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        #  abs(w/2-cx)<w*0.3 and abs(h/2-cy)<h*0.4 : TWEAK, the idea here is to keep only the contours which are closed to the center of the zone
        if area >= min_area and abs(w / 2 - cx) < w * 0.3 and abs(h / 2 - cy) < h * 0.4 and solidity > min_solidity:
            ## DEBUG AREA
            if debug != "no":
                cv2.drawContours(zone, [c], 0, (255, 0, 0), -1)
            if concat_contour is None:
                concat_contour = c
            else:
                concat_contour = np.concatenate((concat_contour, c))
        if debug != "no" and solidity <= min_solidity:
            print("Solidity", solidity)
            cv2.drawContours(strange_cnt, [c], 0, 255, 2)
            cv2.imshow("Strange contours", strange_cnt)

    if concat_contour is not None:
        # At this point, we suppose that 'concat_contour' contains only the contours corresponding the value and suit symbols
        # We can now determine the hull
        hull = cv2.convexHull(concat_contour)
        hull_area = cv2.contourArea(hull)
        # If the area of the hull is to small or too big, there may be a problem
        min_hull_area = 650  # TWEAK, deck and 'zoom' dependant
        max_hull_area = 1500  # TWEAK, deck and 'zoom' dependant
        if hull_area < min_hull_area or hull_area > max_hull_area:
            ok = False
            if debug != "no":
                print("Hull area=", hull_area, "too large or too small")
        # So far, the coordinates of the hull are relative to 'zone'
        # We need the coordinates relative to the image -> 'hull_in_img'
        hull_in_img = hull + corner[0]
    else:
        ok = False

    if debug != "no":
        if concat_contour is not None:
            cv2.drawContours(zone, [hull], 0, (0, 255, 0), 1)
            cv2.drawContours(img, [hull_in_img], 0, (0, 255, 0), 1)
        cv2.imshow("Zone", zone)
        cv2.imshow("Image", img)
        if ok and debug != "pause_always":
            key = cv2.waitKey(1)
        else:
            key = cv2.waitKey(0)
        if key == 27:
            return None
    if ok == False:
        return None
    if test == True:
        cv2.imshow("Zone", zone)
        cv2.imshow("Image", img)
        key = cv2.waitKey(0)
    return hull_in_img


class Cards():
    def __init__(self, cards_pck_fn=cards_pck_fn):
        self._cards = pickle.load(open(cards_pck_fn, 'rb'))
        # self._cards is a dictionary where keys are card names (ex:'Kc') and values are lists of (img,hullHL,hullLR)
        self._nb_cards_by_value = {k: len(self._cards[k]) for k in self._cards}
        print("Nb of cards loaded per name :", self._nb_cards_by_value)

    def get_random(self, card_name=None, display=False):
        if card_name is None:
            card_name = random.choice(list(self._cards.keys()))
        card, hull1 = self._cards[card_name][random.randint(0, self._nb_cards_by_value[card_name] - 1)]
        return card, card_name, hull1


class Backgrounds():
    def __init__(self, backgrounds_pck_fn=backgrounds_pck_fn):
        self._images = pickle.load(open(backgrounds_pck_fn, 'rb'))
        self._nb_images = len(self._images)
        print("Nb of images loaded :", self._nb_images)

    def get_random(self, display=False):
        bg = self._images[random.randint(0, self._nb_images - 1)]
        return bg


def pickle_that():
    dtd_dir = "./dtd/images/"
    bg_images = []
    for subdir in glob(dtd_dir + "/*"):
        for f in glob(subdir + "/*.jpg"):
            bg_images.append(mpimg.imread(f))
    print("Nb of images loaded :", len(bg_images))
    print("Saving in :", backgrounds_pck_fn)
    pickle.dump(bg_images, open(backgrounds_pck_fn, 'wb'))
    print("Save Completed")


def pickle_this():
    imgs_dir = "data/cards"

    cards = {}
    for suit in card_suits:
        for value in card_values:
            card_name = value + suit
            card_dir = os.path.join(imgs_dir, card_name)
            if not os.path.isdir(card_dir):
                print(f"!!! {card_dir} does not exist !!!")
                continue
            cards[card_name] = []
            for f in glob(card_dir + "/*.jpg"):
                img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                hullHL = findHull(img)
                if hullHL is None:
                    print(f"File {f} not used.")
                    continue
                # hullLR = findHull(img, refCornerLR, debug="no")
                # if hullLR is None:
                #     print(f"File {f} not used.")
                #     continue
                # We store the image in "rgb" format (we don't need opencv anymore)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                cards[card_name].append((img, hullHL))
            print(f"Nb images for {card_name} : {len(cards[card_name])}")

    print("Saving in :", cards_pck_fn)
    pickle.dump(cards, open(cards_pck_fn, 'wb'))
    print("Save Completed")


# Code has been modified to only find one hull. - this could be a problem,
# as it means positioning the cards could be harder.


def create_voc_xml(xml_file, img_file, listbba, display=False):
    with open(xml_file, "w") as f:
        f.write(xml_body_1.format(
            **{'FILENAME': os.path.basename(img_file), 'PATH': img_file, 'WIDTH': imgW, 'HEIGHT': imgH}))
        for bba in listbba:
            f.write(xml_object.format(
                **{'CLASS': bba.classname, 'XMIN': bba.x1, 'YMIN': bba.y1, 'XMAX': bba.x2, 'YMAX': bba.y2}))
        f.write(xml_body_2)
        if display: print("New xml", xml_file)


def give_me_filename(dirname, suffixes, prefix=""):
    """
        Function that returns a filename or a list of filenames in directory 'dirname'
        that does not exist yet. If 'suffixes' is a list, one filename per suffix in 'suffixes':
        filename = dirname + "/" + prefix + random number + "." + suffix
        Same random number for all the file name
        Ex:
        > give_me_filename("dir","jpg", prefix="prefix")
        'dir/prefix408290659.jpg'
        > give_me_filename("dir",["jpg","xml"])
        ['dir/877739594.jpg', 'dir/877739594.xml']
    """
    if not isinstance(suffixes, list):
        suffixes = [suffixes]

    suffixes = [p if p[0] == '.' else '.' + p for p in suffixes]

    while True:
        bname = "%09d" % random.randint(0, 999999999)
        fnames = []
        for suffix in suffixes:
            fname = os.path.join(dirname, prefix + bname + suffix)
            if not os.path.isfile(fname):
                fnames.append(fname)

        if len(fnames) == len(suffixes): break

    if len(fnames) == 1:
        return fnames[0]
    else:
        return fnames


###########################################################
"following code taken from jupyter"
# Scenario with 2 cards:
# The original image of a card has the shape (cardH,cardW,4)
# We first paste it in a zero image of shape (imgH,imgW,4) at position decalX, decalY
# so that the original image is centered in the zero image
decalX = int((imgW - cardW) / 2)
decalY = int((imgH - cardH) / 2)

# print("decalX = int((imgW - cardW) / 2)")
# print(decalX)
# print("decalY = int((imgH - cardH) / 2)")
# print(decalY)

# Scenario with 3 cards : decal values are different
decalX3 = int(imgW / 2)
decalY3 = int(imgH / 2 - cardH)


def kps_to_polygon(kps):
    """
        Convert imgaug keypoints to shapely polygon
    """
    pts = [(kp.x, kp.y) for kp in kps]
    return Polygon(pts)


def hull_to_kps(hull, decalX=decalX, decalY=decalY):
    """
        Convert hull to imgaug keypoints
    """
    # hull is a cv2.Contour, shape : Nx1x2
    kps = [ia.Keypoint(x=p[0] + decalX, y=p[1] + decalY) for p in hull.reshape(-1, 2)]
    kps = ia.KeypointsOnImage(kps, shape=(imgH, imgW, 3))
    return kps


def kps_to_BB(kps):
    """
        Determine imgaug bounding box from imgaug keypoints
    """
    extend = 3  # To make the bounding box a little bit bigger
    kpsx = [kp.x for kp in kps.keypoints]
    minx = max(0, int(min(kpsx) - extend))
    maxx = min(imgW, int(max(kpsx) + extend))
    kpsy = [kp.y for kp in kps.keypoints]
    miny = max(0, int(min(kpsy) - extend))
    maxy = min(imgH, int(max(kpsy) + extend))
    if minx == maxx or miny == maxy:
        return None
    else:
        return ia.BoundingBox(x1=minx, y1=miny, x2=maxx, y2=maxy)


# imgaug keypoints of the bounding box of a whole card
cardKP = ia.KeypointsOnImage([
    ia.Keypoint(x=decalX, y=decalY),
    ia.Keypoint(x=decalX + cardW, y=decalY),
    ia.Keypoint(x=decalX + cardW, y=decalY + cardH),
    ia.Keypoint(x=decalX, y=decalY + cardH)
], shape=(imgH, imgW, 3))

# imgaug transformation for one card in scenario with 2 cards
transform_1card = iaa.Sequential([
    iaa.Affine(scale=[0.65, 1]),
    iaa.Affine(rotate=(-180, 180)),
    iaa.Affine(translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)}),
])

# For the 3 cards scenario, we use 3 imgaug transforms, the first 2 are for individual cards,
# and the third one for the group of 3 cards
trans_rot1 = iaa.Sequential([
    iaa.Affine(translate_px={"x": (10, 20)}),
    iaa.Affine(rotate=(22, 30))
])
trans_rot2 = iaa.Sequential([
    iaa.Affine(translate_px={"x": (0, 5)}),
    iaa.Affine(rotate=(10, 15))
])
transform_3cards = iaa.Sequential([
    iaa.Affine(translate_px={"x": decalX - decalX3, "y": decalY - decalY3}),
    iaa.Affine(scale=[0.65, 1]),
    iaa.Affine(rotate=(-180, 180)),
    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
])

# imgaug transformation for the background
scaleBg = iaa.Resize({"height": imgH, "width": imgW})


def augment(img, list_kps, seq, restart=True):
    """
        Apply augmentation 'seq' to image 'img' and keypoints 'list_kps'
        If restart is False, the augmentation has been made deterministic outside the function (used for 3 cards scenario)
    """
    # Make sequence deterministic
    while True:
        if restart:
            myseq = seq.to_deterministic()
        else:
            myseq = seq
        # Augment image, keypoints and bbs
        img_aug = myseq.augment_images([img])[0]

        list_kps_aug = [myseq.augment_keypoints([kp])[0] for kp in list_kps]
        # print("list_kps")
        # print(list_kps)
        # print("list_kps_aug")
        # print(list_kps_aug)
        # print(len(list_kps_aug))
        # foo = (list_kps_aug[1])
        # print("foo")
        # print(foo)
        # bar = (list_kps_aug[2])
        # list_bbs = [kps_to_BB(list_kps_aug[1]), kps_to_BB(list_kps_aug[2])]
        list_bbs = [kps_to_BB(list_kps_aug[1])]

        valid = True
        # Check the card bounding box stays inside the image
        for bb in list_bbs:
            if bb is None or int(round(bb.x2)) >= imgW or int(round(bb.y2)) >= imgH or int(bb.x1) <= 0 or int(
                    bb.y1) <= 0:
                valid = False
                break
        if valid:
            break
        elif not restart:
            img_aug = None
            break

    return img_aug, list_kps_aug, list_bbs


class BBA:  # Bounding box + annotations
    def __init__(self, bb, classname):
        self.x1 = int(round(bb.x1))
        self.y1 = int(round(bb.y1))
        self.x2 = int(round(bb.x2))
        self.y2 = int(round(bb.y2))
        self.classname = classname


class Scene:
    """removing hullb"""

    def __init__(self, bg, img1, class1, hulla1,
                 img2, class2, hulla2,
                 img3=None, class3=None, hulla3=None):
        if img3 is not None:
            self.create3CardsScene(bg, img1, class1, hulla1, img2, class2, hulla2, img3, class3, hulla3)
        else:
            self.create2CardsScene(bg, img1, class1, hulla1, img2, class2, hulla2)

    def create2CardsScene(self, bg, img1, class1, hulla1,
                          img2, class2, hulla2):
        kpsa1 = hull_to_kps(hulla1)
        kpsa2 = hull_to_kps(hulla2)

        # Randomly transform 1st card
        self.img1 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img1.fill(255)
        # cv2.imshow("self.img", self.img1)
        # cv2.imshow("img1", img1)
        # cv2.waitKey(0)4
        # print(img1.shape)
        # print(self.img1.shape)

        self.img1[decalY:(decalY + cardH * 4), decalX:(decalX + cardW * 4), :] = img1

        # self.img1[0:cardH*4, 0:cardW*4, :] = img1

        # cv2.imshow("img", self.img1)
        # cv2.waitKey(0)

        self.img1, self.lkps1, self.bbs1 = augment(self.img1, [cardKP, kpsa1], transform_1card)

        # Randomly transform 2nd card. We want that card 2 does not partially cover a corner of 1 card.
        # If so, we apply a new random transform to card 2
        while True:
            self.listbba = []
            self.img2 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
            self.img2[decalY:decalY + cardH * 4, decalX:decalX + cardW * 4, :] = img2
            self.img2, self.lkps2, self.bbs2 = augment(self.img2, [cardKP, kpsa2], transform_1card)

            # mainPoly2: shapely polygon of card 2
            mainPoly2 = kps_to_polygon(self.lkps2[0].keypoints[0:4])
            invalid = False
            intersect_ratio = 0.1
            for i in range(1, 2):
                # smallPoly1: shapely polygon of one of the hull of card 1
                smallPoly1 = kps_to_polygon(self.lkps1[i].keypoints[:])
                a = smallPoly1.area
                # We calculate area of the intersection of card 1 corner with card 2
                intersect = mainPoly2.intersection(smallPoly1)
                ai = intersect.area
                # If intersection area is small enough, we accept card 2
                if (a - ai) / a > 1 - intersect_ratio:
                    self.listbba.append(BBA(self.bbs1[i - 1], class1))
                # If intersectio area is not small, but also not big enough, we want apply new transform to card 2
                elif (a - ai) / a > intersect_ratio:
                    invalid = True
                    break

            if not invalid: break

        self.class1 = class1
        self.class2 = class2
        for bb in self.bbs2:
            self.listbba.append(BBA(bb, class2))
        # Construct final image of the scene by superimposing: bg, img1 and img2
        self.bg = scaleBg.augment_image(bg)
        mask1 = self.img1[:, :, 3]
        self.mask1 = np.stack([mask1] * 3, -1)
        self.final = np.where(self.mask1, self.img1[:, :, 0:3], self.bg)
        mask2 = self.img2[:, :, 3]
        self.mask2 = np.stack([mask2] * 3, -1)
        self.final = np.where(self.mask2, self.img2[:, :, 0:3], self.final)

    def create3CardsScene(self, bg, img1, class1, hulla1, img2, class2, hulla2, img3, class3, hulla3):

        kpsa1 = hull_to_kps(hulla1, decalX3, decalY3)
        kpsa2 = hull_to_kps(hulla2, decalX3, decalY3)
        kpsa3 = hull_to_kps(hulla3, decalX3, decalY3)
        self.img3 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img3[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img3
        self.img3, self.lkps3, self.bbs3 = augment(self.img3, [cardKP, kpsa3], trans_rot1)
        self.img2 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img2[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img2
        self.img2, self.lkps2, self.bbs2 = augment(self.img2, [cardKP, kpsa2], trans_rot2)
        self.img1 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img1[decalY3:decalY3 + cardH, decalX3:decalX3 + cardW, :] = img1

        while True:
            det_transform_3cards = transform_3cards.to_deterministic()
            _img3, _lkps3, self.bbs3 = augment(self.img3, self.lkps3, det_transform_3cards, False)
            if _img3 is None: continue
            _img2, _lkps2, self.bbs2 = augment(self.img2, self.lkps2, det_transform_3cards, False)
            if _img2 is None: continue
            _img1, self.lkps1, self.bbs1 = augment(self.img1, [cardKP, kpsa1], det_transform_3cards, False)
            if _img1 is None: continue
            break
        self.img3 = _img3
        self.lkps3 = _lkps3
        self.img2 = _img2
        self.lkps2 = _lkps2
        self.img1 = _img1

        self.class1 = class1
        self.class2 = class2
        self.class3 = class3
        self.listbba = [BBA(self.bbs1[0], class1), BBA(self.bbs2[0], class2), BBA(self.bbs3[0], class3),
                        BBA(self.bbs3[1], class3)]

        # Construct final image of the scene by superimposing: bg, img1, img2 and img3
        self.bg = scaleBg.augment_image(bg)
        mask1 = self.img1[:, :, 3]
        self.mask1 = np.stack([mask1] * 3, -1)
        self.final = np.where(self.mask1, self.img1[:, :, 0:3], self.bg)
        mask2 = self.img2[:, :, 3]
        self.mask2 = np.stack([mask2] * 3, -1)
        self.final = np.where(self.mask2, self.img2[:, :, 0:3], self.final)
        mask3 = self.img3[:, :, 3]
        self.mask3 = np.stack([mask3] * 3, -1)
        self.final = np.where(self.mask3, self.img3[:, :, 0:3], self.final)

    def display(self):
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(self.final)
        for bb in self.listbba:
            rect = patches.Rectangle((bb.x1, bb.y1), bb.x2 - bb.x1, bb.y2 - bb.y1, linewidth=1, edgecolor='b',
                                     facecolor='none')
            ax.add_patch(rect)

    def res(self):
        return self.final

    def write_files(self, save_dir, display=False):
        jpg_fn, xml_fn = give_me_filename(save_dir, ["jpg", "xml"])
        plt.imsave(jpg_fn, self.final)
        if display: print("New image saved in", jpg_fn)
        create_voc_xml(xml_fn, jpg_fn, self.listbba, display=display)


def generate_scenes(backgrounds, cards, n):
    nb_cards_to_generate = n
    save_dir = "data/scenes/val"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for i in tqdm(range(nb_cards_to_generate)):
        bg = backgrounds.get_random()
        img1, card_val1, hulla1 = cards.get_random()
        img2, card_val2, hulla2 = cards.get_random()

        newimg = Scene(bg, img1, card_val1, hulla1,
                       img2, card_val2, hulla2)
        newimg.write_files(save_dir)


def main():
    print("main")
    # first place cards in dataset2_blackbackground/
    # with cardnumbersuit name
    # choose a few cards to adjust settings with.
    # card_prep2("./dataset2_blackbackground/2c.jpg")
    # save acceptable settings
    # extract_all("./dataset2_blackbackground/")
    # create varying brightness and contrast
    # augment_images(30)
    # adjust cards now to see if the hull areas fit
    # adjust the area values in findhull  function if they are too small
    # imghull = cv2.imread("./data/cards/10c/10c.jpg")
    # cv2.imshow("orig", imghull)
    # findHull(imghull, debug=True,test=True)

    # create pickle files of the background and the cards
    # N.B. backgrounds must be downloaded from here:
    # wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
    # tar xf dtd-r1.0.1.tar.gz dtd

    # pickle_that()
    # pickle_this()
    # cards = Cards()
    # backgrounds = Backgrounds()

    # test card scene generation
    # bg = backgrounds.get_random()
    # img1, card_val1, hulla1= cards.get_random()
    # img2, card_val2, hulla2= cards.get_random()
    # #
    # newimg = Scene(bg, img1, card_val1, hulla1,
    #                      img2, card_val2, hulla2)
    # print("image created")
    # newimg.write_files()

    # generate scenes
    # generate_scenes(backgrounds, cards, 5000)

    ## In case you want to train YOLO with the generated datasets
    # YOLO cannot directly exploit the Pascal VOC annotations files.
    # You need to convert the xml files in txt files accordingly to the syntax explained here:
    # https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
    # The script 'convert_voc_yolo.py' makes this conversion and also generates
    # the txt file that contains all the images of the dataset

    # !python convert_voc_yolo.py data/scenes/val data/cards.names data/val.txt
    # python convert_voc_yolo.py data/scenes/train data/cards.names data/train.txt


main()






