import argparse
import cv2

import correct_perspective, findboxes3

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

    # detect and display lines and boxes
    findboxes3.findboxes(corrected_image, args.verbose)

    # clean up
    if args.verbose >= 1:
        cv2.waitKey(0)
        cv2.destroyAllWindows()