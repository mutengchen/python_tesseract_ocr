
import cv2
import numpy as np
def locater(image, source, num=0):
    def resize(im, new_width):
        r = float(new_width) / im.shape[1]
        dim = (new_width, int(im.shape[0] * r))
        return cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    #width = 300
    #source = resize(source, new_width=width)
    #image = resize(image, new_width=width)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    image, u, v = cv2.split(hsv)

    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2LUV)
    source, u, v = cv2.split(hsv)

    MIN_MATCH_COUNT = 10
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image, None)
    kp2, des2 = orb.detectAndCompute(source, None)

    flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

    des1 = np.asarray(des1, dtype=np.float32)
    des2 = np.asarray(des2, dtype=np.float32)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h,w = image.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        source_bgr = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        img2 = cv2.polylines(source_bgr, [np.int32(dst)], True, (0,0,255), 3, 
                             cv2.LINE_AA)
        cv2.imwrite("out"+str(num)+".jpg", img2)
    else:
        print("Not enough matches." + str(len(good)))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0), # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask, # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(image, kp1, source, kp2, good, None, **draw_params)
    cv2.imwrite("ORB"+str(num)+".jpg", img3)

image = cv2.imread('c3.png')
source = cv2.imread('c3.png')
locater(source, image, num=1)