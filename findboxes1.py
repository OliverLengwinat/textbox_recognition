import cv2
import argparse

parser = argparse.ArgumentParser(description='Trying to detect boxes on an image.')
parser.add_argument('--imgloc', '-i', default='images/1_300_400.png', help='the (single) input image\'s location')
args = parser.parse_args()

image = cv2.imread(args.imgloc)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 50, 200, 1)

cnts = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
print('{} contours found'.format(len(cnts)))
for c in cnts:
    area = cv2.contourArea(c)
    if area > 600 and area < 1200:
        cv2.drawContours(image, [c], 0, (36,255,12), 2)

cv2.imshow('canny', canny)
cv2.imshow('image', image)
cv2.waitKey()
