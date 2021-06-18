import cv2
import numpy as np
import utils

utils.DISP = True

try:
    template = cv2.imread("img/template.png")
    test = cv2.imread("img/test_p.jpg")

    utils.disp(template)
    utils.disp(test)

    matches_img, dst = utils.match_images(template, test)

    utils.disp(matches_img)

    cropped_masked = utils.mask_and_crop(test, dst)

    utils.disp(cropped_masked)

    color_masked, color_mask, contours = utils.find_lines(cropped_masked)

    utils.disp(color_masked)
    utils.disp(color_mask, wait=True)

    if len(contours) == 1:
        print("Test is negative :)")
    elif len(contours) == 2:
        print("Test is positive :(")
    else:
        print("Bad test image, try with a different one")
except:
    print("Bad test image, try with a different one")