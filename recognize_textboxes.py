import argparse
import cv2
from os import listdir, remove

import correct_perspective, findboxes, mnist_preprocessing, train_mnist

if __name__ == '__main__':
    # arguments parsing
    parser = argparse.ArgumentParser(description='Trying to detect boxes on an image.')
    parser.add_argument('--imgloc', '-i', default='images/1_notear_600_800.png', help='the (single) input image\'s location')
    parser.add_argument('--verbose', '-v', help='increase output verbosity', action='count', default=0) 
    args = parser.parse_args()

    image = cv2.imread(args.imgloc)
    if args.verbose >= 1:
        cv2.imshow('original image', image)

    # correct skew/perspective
    corrected_image = correct_perspective.correct_skew(image, args.verbose)
    if args.verbose >= 1:
        cv2.imshow("corrected image", corrected_image)

    # detect and display lines and boxes (saved to harddrive)
    detected_digits = findboxes.findboxes(corrected_image, args.verbose)

    # MNIST preprocessing
    # delete previous folder contents
    files = listdir("output_digits")
    for f in files:
        remove("output_digits/"+f)

    for field_idx, field in enumerate(detected_digits):
        for digit_idx, digit in enumerate(field):
            digit_img_mnist = mnist_preprocessing.mnist_preprocessing(digit)
            if args.verbose >=2:
                cv2.imshow("field "+str(field_idx+1)+", digit "+str(digit_idx+1), digit_img_mnist)
            cv2.imwrite("output_digits/field_"+str(field_idx+1)+"_digit_"+str(digit_idx+1)+".png", digit_img_mnist)

    # get MNIST trained network
    predictions = train_mnist.train_and_predict(listdir("output_digits"), verbosity=args.verbose)

    for prediction, image_name in zip(predictions, listdir("output_digits")):
        current_image = cv2.imread("output_digits/"+image_name)
        cv2.imshow(image_name+" - prediction: "+str(prediction), current_image)
    cv2.waitKey()

    # clean up
    if args.verbose >= 1:
        cv2.waitKey(0)
        cv2.destroyAllWindows()