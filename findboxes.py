# binarization process from https://stackoverflow.com/a/57103789
# box detection from https://medium.com/coinmonks/a-box-detection-algorithm-for-any-image-containing-boxes-756c15d7ed26

import argparse
import cv2
from math import ceil, floor
import numpy as np

# minimum size of extracted boxes
MIN_WIDTH = 0.04
MAX_WIDTH = 0.10
MIN_HEIGHT = 0.02
MAX_HEIGHT = 0.04
MIN_ASP_RATIO = 1.8
MAX_ASP_RATIO = 3.0

# minimum length of straight lines to extract (unit unknown)
MIN_EDGE_LENGTH = 4

# These are probably the only important parameters in the
# whole pipeline (steps 0 through 3).
BLOCK_SIZE = 40
DELTA = 25

# Do the necessary noise cleaning and other stuffs.
# I just do a simple blurring here but you can optionally
# add more stuffs.
def preprocess(image):
    image = cv2.medianBlur(image, 3)
    return 255 - image

# Again, this step is fully optional and you can even keep
# the body empty. I just did some opening. The algorithm is
# pretty robust, so this stuff won't affect much.
def postprocess(image):
    kernel = np.ones((3,3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image

def get_color_mask(image):
    # make mask and inverted mask for colored areas ToDo: remove color part?
    b,g,r = cv2.split(cv2.blur(image,(5,5)))
    np.seterr(divide='ignore', invalid='ignore') # 0/0 --> 0
    m = (np.fmin(np.fmin(b, g), r) / np.fmax(np.fmax(b, g), r)) * 255
    _,mask_inv = cv2.threshold(np.uint8(m), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(mask_inv)
    return mask

# Just a helper function that generates box coordinates
def get_block_index(image_shape, yx, block_size):
    y = np.arange(max(0, yx[0]-block_size), min(image_shape[0], yx[0]+block_size))
    x = np.arange(max(0, yx[1]-block_size), min(image_shape[1], yx[1]+block_size))
    return np.meshgrid(y, x)

# Here is where the trick begins. We perform binarization from the
# median value locally (the img_in is actually a slice of the image).
# Here, following assumptions are held:
#   1.  The majority of pixels in the slice is background
#   2.  The median value of the intensity histogram probably
#       belongs to the background. We allow a soft margin DELTA
#       to account for any irregularities.
#   3.  We need to keep everything other than the background.
#
# We also do simple morphological operations here. It was just
# something that I empirically found to be "useful", but I assume
# this is pretty robust across different datasets.
def adaptive_median_threshold(img_in):
    med = np.median(img_in)
    img_out = np.zeros_like(img_in)
    img_out[img_in - med < DELTA] = 255
    kernel = np.ones((3,3),np.uint8)
    img_out = 255 - cv2.dilate(255 - img_out,kernel,iterations = 2)
    return img_out

# This function just divides the image into local regions (blocks),
# and perform the `adaptive_mean_threshold(...)` function to each
# of the regions.
def block_image_process(image, block_size):
    out_image = np.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[tuple(block_idx)] = adaptive_median_threshold(image[tuple(block_idx)])
    return out_image

# This function invokes the whole pipeline of Step 2.
def process_image(img):
    image_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_in = preprocess(image_in)
    image_out = block_image_process(image_in, BLOCK_SIZE)
    image_out = postprocess(image_out)
    return image_out

# This is the function used for composing
def sigmoid(x, orig, rad):
    k = np.exp((x - orig) * 5 / rad)
    return k / (k + 1.)

# Here, we combine the local blocks. A bit lengthy, so please
# follow the local comments.
def combine_block(img_in, mask):
    # First, we pre-fill the masked region of img_out to white
    # (i.e. background). The mask is retrieved from previous section.
    img_out = np.zeros_like(img_in)
    img_out[mask == 255] = 255
    fimg_in = img_in.astype(np.float32)

    # Then, we store the foreground (letters written with ink)
    # in the `idx` array. If there are none (i.e. just background),
    # we move on to the next block.
    idx = np.where(mask == 0)
    if idx[0].shape[0] == 0:
        img_out[idx] = img_in[idx]
        return img_out

    # We find the intensity range of our pixels in this local part
    # and clip the image block to that range, locally.
    lo = fimg_in[idx].min()
    hi = fimg_in[idx].max()
    v = fimg_in[idx] - lo
    r = hi - lo

    # Now we use good old OTSU binarization to get a rough estimation
    # of foreground and background regions.
    img_in_idx = img_in[idx]
    _ret3, th3 = cv2.threshold(img_in[idx],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Then we normalize the stuffs and apply sigmoid to gradually
    # combine the stuffs.
    bound_value = np.min(img_in_idx[th3[:, 0] == 255])
    bound_value = (bound_value - lo) / (r + 1e-5)
    f = (v / (r + 1e-5))
    f = sigmoid(f, bound_value + 0.05, 0.2)

    # Finally, we re-normalize the result to the range [0..255]
    img_out[idx] = (255. * f).astype(np.uint8)
    return img_out

# We do the combination routine on local blocks, so that the scaling
# parameters of Sigmoid function can be adjusted to local setting
def combine_block_image_process(image, mask, block_size):
    out_image = np.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[tuple(block_idx)] = combine_block(
                image[tuple(block_idx)], mask[tuple(block_idx)])
    return out_image

# Postprocessing (should be robust even without it, but I recommend
# you to play around a bit and find what works best for your data.
# I just left it blank.
def combine_postprocess(image):
    return image

# The main function of this section. Executes the whole pipeline.
def combine_process(img, mask):
    image_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_out = combine_block_image_process(image_in, mask, 20)
    image_out = combine_postprocess(image_out)
    return image_out


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

# the main function
def findboxes(image, verbosity=0):
    mask = get_color_mask(image)
    image_highcontrast = combine_process(image, mask)

    if verbosity >= 2:
        cv2.imshow("image_highcontrast", image_highcontrast)

    # Thresholding the image
    _thresh, img_bin = cv2.threshold(image_highcontrast, 128, 255,cv2.THRESH_BINARY|     cv2.THRESH_OTSU)

    # Invert the image
    img_bin = 255-img_bin
    if verbosity >= 2:
        cv2.imshow("Image_bin",img_bin)

    # Defining a kernel length
    kernel_length = np.array(image).shape[1]//80*MIN_EDGE_LENGTH

    # A vertical kernel of (1 X kernel_length), which will detect all the vertical lines from the image.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
    # dilation is done in more iterations than erosion to increase line length
    vertical_lines_img = cv2.dilate(img_temp1, vertical_kernel, iterations=7)
    if verbosity >= 2:
        cv2.imshow("vertical_lines.jpg",vertical_lines_img)
    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=4)
    if verbosity >= 2:
        cv2.imshow("horizontal_lines.jpg",horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_binary_boxes = cv2.addWeighted(vertical_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_binary_boxes = cv2.erode(~img_binary_boxes, kernel, iterations=2)
    _thresh, img_binary_boxes = cv2.threshold(img_binary_boxes, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if verbosity >= 1:
        cv2.imshow("img_binary_boxes", img_binary_boxes)

    # Find contours for image, which will detect all the boxes
    contours, _hierarchy = cv2.findContours(img_binary_boxes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, _boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    #img_annotated_boxes = img_binary_boxes
    img_annotated_boxes = image
    idx = 0
    for c in contours:
        # Return the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)

        # Only save the box in output folder if it meets these dimension criteria
        if (floor(MIN_WIDTH*image.shape[1]) < w < ceil(MAX_WIDTH*image.shape[1]) and 
            floor(MIN_HEIGHT*image.shape[0]) < h < ceil(MAX_HEIGHT*image.shape[0]) and 
            MIN_ASP_RATIO < w/h < MAX_ASP_RATIO):
            idx += 1
            new_img = image[y:y+h, x:x+w]
            new_img_hc = image_highcontrast[y:y+h, x:x+w]
            cv2.imwrite('output/'+str(idx) + '.png', new_img)
            cv2.imwrite('output_high_contrast/'+str(idx) + '.png', new_img_hc)

            color = (255, 0, 0)
            cv2.rectangle(img_annotated_boxes, (x,y), (x+w,y+h), color, 1)
            cv2.putText(img_annotated_boxes, str(idx), (x,y+h), cv2.FONT_HERSHEY_PLAIN, 0.8, color)
    
    print("{} boxes detected (of 114)".format(idx))
    if verbosity >= 1:
        cv2.imshow("annotated boxes", img_annotated_boxes)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trying to detect boxes on an image.')
    parser.add_argument('--imgloc', '-i', default='images/1_300_400.png', help='the (single) input image\'s location')
    parser.add_argument('--verbose', '-v', help='increase output verbosity', action='count', default=0)    
    args = parser.parse_args()
    input_image = cv2.imread(args.imgloc)

    findboxes(input_image, args.verbose)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
