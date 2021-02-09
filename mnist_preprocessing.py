# source: https://opensourc.es/blog/tensorflow-mnist/ / https://github.com/opensourcesblog/tensorflow-mnist/blob/master/learn_extra.py

import cv2
import math
import numpy as np
from scipy import ndimage

def resize(gray):
    return cv2.resize(gray, (28, 28))

def remove_empty_spaces(gray):
    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)
    
    return gray

def resize_to_box(gray):
    # MNIST expects the number inside the image to be 20x20

    rows, cols = gray.shape

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        gray = cv2.resize(gray, (cols, rows))

    return gray

def enhance_to_box(gray):
    # MNIST expects images of 28x28 (including padding)

    rows, cols = gray.shape

    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

    return gray

def getBestShift(img):
    # helper function to shift by weight
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    # shift image by sx, sy
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def shift_to_center_of_weight(gray):
    # shift number by weight as was done in MNIST training
    shiftx, shifty = getBestShift(gray)
    shifted = shift(gray, shiftx, shifty)
    return shifted

def mnist_preprocessing(img):
    # combination of preprocessing steps

    # img = resize(img)
    img = remove_empty_spaces(img)
    img = resize_to_box(img)
    img = enhance_to_box(img)
    img = shift_to_center_of_weight(img)

    return img


def flatten(img):
    # change shape to array as expected by classifier
    return img.flatten() / 255.0
