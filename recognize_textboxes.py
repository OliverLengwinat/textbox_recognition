import argparse
import cv2

import correct_perspective

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trying to detect boxes on an image.')
    parser.add_argument('--imgloc', '-i', default='images/1_notear_300_400.png', help='the (single) input image\'s location')
    parser.add_argument('--verbose', '-v', help='increase output verbosity', action='store_true')    
    args = parser.parse_args()
    image = cv2.imread(args.imgloc)
    cv2.imshow('original image', image)

    corrected_image = correct_perspective.correct_skew(image, args.verbose)

    cv2.imshow("corrected image", corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()