# partially taken from https://github.com/ashuta03/automatic_skew_correction_using_corner_detectors_and_homography/tree/master#

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_destination_points(corners):
    """
    -Get destination points from corners of warped images
    -Approximating height and width of the rectangle: we take maximum of the 2 widths and 2 heights

    Args:
        corners: list

    Returns:
        destination_corners: list
        height: int
        width: int

    """

    w1 = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
    w2 = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
    w = max(int(w1), int(w2))

    h1 = np.sqrt((corners[0][0] - corners[2][0]) ** 2 + (corners[0][1] - corners[2][1]) ** 2)
    h2 = np.sqrt((corners[1][0] - corners[3][0]) ** 2 + (corners[1][1] - corners[3][1]) ** 2)
    h = max(int(h1), int(h2))

    destination_corners = np.float32([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])

    print('\nThe destination points are: \n')
    for index, c in enumerate(destination_corners):
        character = chr(65 + index) + "'"
        print(character, ':', c)

    print('\nThe approximated height and width of the original image is: \n', (h, w))
    return destination_corners, h, w

def unwarp(img, src, dst):
    """

    Args:
        img: np.array
        src: list
        dst: list

    Returns:
        un_warped: np.array

    """
    h, w = img.shape[:2]
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    print('\nThe homography matrix is: \n', H)
    un_warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)

    return un_warped

def apply_filter(image):
    """
    Define a 5X5 kernel and apply the filter to gray scale image
    Args:
        image: np.array

    Returns:
        filtered: np.array

    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.float32) / 15
    filtered = cv2.filter2D(gray, -1, kernel)
    
    return filtered

def apply_threshold(filtered):
    """
    Apply OTSU threshold

    Args:
        filtered: np.array

    Returns:
        thresh: np.array

    """
    _, thresh = cv2.threshold(filtered, 250, 255, cv2.THRESH_OTSU)

    return thresh

def detect_contour(img, image_shape):
    """

    Args:
        img: np.array()
        image_shape: tuple

    Returns:
        canvas: np.array()
        cnt: list

    """
    canvas = np.zeros(image_shape, np.uint8)
    contours, _hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(canvas, cnt, -1, (0, 255, 255), 3)

    return canvas, cnt

def detect_corners_from_contour(canvas, cnt):
    """
    Detecting corner points form contours using cv2.approxPolyDP()
    Args:
        canvas: np.array()
        cnt: list

    Returns:
        approx_corners: list

    """
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
    approx_corners = sorted(np.concatenate(approx_corners).tolist())
    print('\nThe corner points are ...\n')
    for index, c in enumerate(approx_corners):
        character = chr(65 + index)
        print(character, ':', c)
        cv2.putText(canvas, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Rearranging the order of the corner points
    approx_corners = [approx_corners[i] for i in [0, 2, 1, 3]]

    return approx_corners, canvas

<<<<<<< HEAD
def correct_skew(image, verbose=False):
=======
def correct_skew(image, verbose):
>>>>>>> 3ed80b71d021e9594f8c6b0f59d6d5e2f1b4ea2f
    """
    Skew correction using homography and corner detection using contour points
    Returns: corrected image

    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    filtered_image = apply_filter(image)
    if verbose:
        plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
        plt.title('Filtered Image')
        plt.show()
    threshold_image = apply_threshold(filtered_image)
    if verbose:
        plt.imshow(cv2.cvtColor(threshold_image, cv2.COLOR_BGR2RGB))
        plt.title('After applying OTSU threshold')
        plt.show()

    cnv, largest_contour = detect_contour(threshold_image, image.shape)
    if verbose:
        plt.title('Largest Contour')
        plt.imshow(cnv)
        plt.show()
    corners, canvas = detect_corners_from_contour(cnv, largest_contour)
    
    if verbose:
        plt.imshow(canvas)
        plt.title('Corner Points: Douglas-Peucker')
        plt.show()

    destination_points, h, w = get_destination_points(corners)

<<<<<<< HEAD
    src = np.float32(corners)
    un_warped = unwarp(image, src, destination_points)
=======
    src_corners = np.float32(corners)
    un_warped = unwarp(image, src_corners, destination_points)
>>>>>>> 3ed80b71d021e9594f8c6b0f59d6d5e2f1b4ea2f
    if verbose:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        # f.subplots_adjust(hspace=.2, wspace=.05)
        ax1.imshow(image)
        ax1.set_title('Original Image')
<<<<<<< HEAD
        x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
        y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
=======
        x = [src_corners[0][0], src_corners[2][0], src_corners[3][0], src_corners[1][0], src_corners[0][0]]
        y = [src_corners[0][1], src_corners[2][1], src_corners[3][1], src_corners[1][1], src_corners[0][1]]
>>>>>>> 3ed80b71d021e9594f8c6b0f59d6d5e2f1b4ea2f
        ax2.imshow(image)
        ax2.plot(x, y, color='yellow', linewidth=3)
        h, w = image.shape[:2]
        ax2.set_ylim([h, 0])
        ax2.set_xlim([0, w])
        ax2.set_title('Target Area')

        plt.show()

    cropped = un_warped[0:h, 0:w]
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    # f.subplots_adjust(hspace=.2, wspace=.05)
    if verbose:
        ax1.imshow(un_warped)
        ax2.imshow(cropped)
        plt.show()

    return cropped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trying to detect boxes on an image.')
    parser.add_argument('--imgloc', '-i', default='images/1_notear_300_400.png', help='the (single) input image\'s location')
    parser.add_argument('--verbose', '-v', help='increase output verbosity', action='store_true')    
    args = parser.parse_args()
    image = cv2.imread(args.imgloc)
    cv2.imshow('original image', image)
    
    corrected_image = correct_skew(image, args.verbose)
    cv2.imshow("corrected image", corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()