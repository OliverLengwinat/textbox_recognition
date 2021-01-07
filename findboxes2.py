import cv2
import numpy as np
import argparse



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
    cv2.imshow("after preprocess", image_in)
    image_out = block_image_process(image_in, BLOCK_SIZE)
    cv2.imshow("after block image process", image_out)
    image_out = postprocess(image_out)
    return image_out



# Somehow I found the value of `gamma=1.2` to be the best in my case
def adjust_gamma(image, gamma=1.2):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

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


#Functon for extracting the box
def box_extraction(img_for_box_extraction_path, cropped_dir_path):

    print("Reading image..")
    img = cv2.imread(img_for_box_extraction_path, cv2.IMREAD_GRAYSCALE)  # Read the image
    cv2.imshow("img", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # call addWeighted function. use beta = 0 to effectively only operate one one image
    contrast = 2
    brightness = 10
    out = adjust_gamma(img)
    cv2.imshow("out", out)


    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin  # Invert the image
    print("Storing binary image to images/Image_bin.jpg..")
    cv2.imwrite("images/Image_bin.jpg",img_bin)
    cv2.imshow("img_bin", img_bin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Applying Morphological Operations..")
    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//40
     
    # A vertical kernel of (1 X kernel_length), which will detect all the vertical lines from the image.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(img_temp1, vertical_kernel, iterations=3)
    cv2.imwrite("images/vertical_lines.jpg",vertical_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite("images/horizontal_lines.jpg",horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(vertical_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # For Debugging
    # Enable this line to see vertical and horizontal lines in the image which is used to find boxes
    print("Binary image which only contains boxes: images/img_final_bin.jpg")
    cv2.imwrite("images/img_final_bin.jpg",img_final_bin)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    print("Output stored in Output directory!")

    idx = 0
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)

        # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
        if (w > 80 and h > 20) and w > 3*h:
            idx += 1
            new_img = img[y:y+h, x:x+w]
            cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)

    # For Debugging
    # Enable this line to see all contours.
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    # cv2.imwrite("./Temp/img_contour.jpg", img)


parser = argparse.ArgumentParser(description='Trying to detect boxes on an image.')
parser.add_argument('--imgloc', '-i', default='images/1_300_400.png', help='the (single) input image\'s location')
args = parser.parse_args()
#image = cv2.imread(args.imgloc)

#Input image path and out folder
#box_extraction(args.imgloc, "./output/")
image_result = process_image(cv2.imread(args.imgloc))
cv2.imshow("result", image_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

