import cv2
import numpy as np
import inspect

MIN_MATCH_COUNT = 10
DISP = True

def get_name(var):
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]

def disp(image, wait = False):
    if(DISP):
        cv2.imshow(get_name(image), image)
        cv2.waitKey(int(not wait))

def match_images(template, test):
    sift = cv2.SIFT_create()

    kps = {}
    des = {}
    kps['template'], des['template'] = sift.detectAndCompute(template,None)
    kps['test'], des['test'] = sift.detectAndCompute(test,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des['template'],des['test'],k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kps['template'][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kps['test'][m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,d = template.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        test = cv2.polylines(test,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), 
                    singlePointColor = None,
                    matchesMask = matchesMask, 
                    flags = 2)
    matches_img = cv2.drawMatches(template,kps['template'],test,kps['test'],good,None,**draw_params)

    return matches_img, dst

def mask_and_crop(test, dst):
    dst = [i[0] for i in dst]
    dst = [list(map(int, i)) for i in dst]
    dst = np.asarray(dst)
    mask = np.zeros(test.shape[:2], np.uint8)
    cv2.drawContours(mask, [dst], -1, (255, 255, 255), -1, cv2.LINE_AA)
    masked = cv2.bitwise_and(test, test, mask=mask)

    rect = cv2.boundingRect(dst)
    x,y,w,h = rect
    cropped_masked = masked[y:y+h, x:x+w].copy()
    return cropped_masked

def find_lines(cropped_masked):
    cropped_masked_hsv=cv2.cvtColor(cropped_masked, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(cropped_masked_hsv, (150,70,50), (180,255,255))
    kernel = np.ones((3, 3), 'uint8')
    color_mask = cv2.dilate(color_mask, kernel, iterations=1)
    color_masked = cv2.bitwise_and(cropped_masked, cropped_masked, mask = color_mask)
    
    contours,h = cv2.findContours(color_mask,1,2)
    for cnt in contours:
        cv2.drawContours(color_masked,[cnt],0,(0,0,255),1)
    
    return color_masked, color_mask, contours