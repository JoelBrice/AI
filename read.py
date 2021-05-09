import cv2 as cv

img = cv.imread('Photos/user.jpg')

cv.imshow('User', img)

cv.wait()