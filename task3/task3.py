import cv2 as cv
import numpy as np

image = cv.imread('rov1.png')


def NameOfImage(img, name):
    img = cv.putText(img, name, (75, 75), cv.FONT_HERSHEY_PLAIN, .9, (0, 255, 0), 2, cv.LINE_AA)
    return img


point1 = None
point2 = None
segment = None


def partition(event, x, y, flags, param):
    global point1, point2, segment
    if event == cv.EVENT_LBUTTONDBLCLK:
        if point1 is None:
            point1 = (x, y)
        elif point2 is None:
            point2 = (x, y)

            minY = min(point1[1], point2[1])
            maxY = max(point1[1], point2[1])
            if minY != maxY:
                segment = image[minY:maxY, :]
            point1 = None
            point2 = None


cv.namedWindow('image')
cv.setMouseCallback('image', partition)


def checkRedStar(img):
    # lower mask
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv.inRange(img_hsv, lower_red, upper_red)
    # Upper mask
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv.inRange(img_hsv, lower_red, upper_red)
    mask = mask0 + mask1
    count = np.count_nonzero(mask)
    if count >= 5000:
        return True
    return False


def checkYellowSquare(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    u = np.uint8([[[0, 236, 236]]])
    lower_yellow = np.array(cv.cvtColor(1, cv.COLOR_BGR2HSV))
    upper_yellow = np.array(cv.cvtColor(u, cv.COLOR_BGR2HSV))
    mask = cv.inRange(img_hsv, lower_yellow, upper_yellow)
    count = np.count_nonzero(mask)
    if count >= 5000:
        return True
    return False


try:
    while(1):
        cv.imshow('image', image)
        if segment is not None:
            cv.imshow('segment', segment)
            left = segment[:, :(int(segment.shape[1] / 3))]
            middle = segment[:, (int(segment.shape[1] / 3)): (int(segment.shape[1] / 3)) * 2]
            right = segment[:, (int(segment.shape[1] / 3)) * 2:]
            if checkRedStar(left):
                left = NameOfImage(left, "Red Star visible")
            if checkRedStar(middle):
                middle = NameOfImage(middle, "Red Star visible")
            if checkRedStar(right):
                right = NameOfImage(right, "Red Star visible")

            if checkYellowSquare(left):
                left = NameOfImage(left, "Yellow Square visible")
            if checkYellowSquare(middle):
                middle = NameOfImage(middle, "Yellow Square visible")
            if checkYellowSquare(right):
                right = NameOfImage(right, "Yellow Square visible")

            cv.imshow('Square 1', left)
            cv.imshow('Square 2', middle)
            cv.imshow('Sqaure 3', right)

        if cv.waitKey(1) & 0xFF == 27:
            break
except Exception as e:
    print(e)
cv.destroyAllWindows()