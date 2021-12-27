import cv2
import os
import numpy as np

MODE = 1 #0表示普通识别，1表示算法优化过的
#使用canny对边缘进行加强


def order_points(pts):
    # 初始化坐标点
    rect = np.zeros((4, 2), dtype = "float32")
    # 获取左上角和右下角坐标点
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 分别计算左上角和右下角的离散差值
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


#仿射变换，把倾斜的图片转换成水平
def four_point_transform(image, pts):

    # 获取坐标点，并将它们分离开来
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # 计算新图片的宽度值，选取水平差值的最大值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))

    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))

    # 计算新图片的高度值，选取垂直差值的最大值

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))

    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxHeight = max(int(heightA), int(heightB))

    # 构建新图片的4个坐标点

    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")

    # 获取仿射变换矩阵并应用它

    M = cv2.getPerspectiveTransform(rect, dst)

    # 进行仿射变换

    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后的结果

    return warped


def start(mode):
    img = cv2.imread('shiji.png')
    img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode==1 :
        canny=cv2.Canny(gray,30,140,3)
        #将边缘增强过的图片的外部轮廓提取出来
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxarea=0
        maxint=0
        i = 0
        for c in contours:
            if cv2.contourArea(c) > maxarea:
                maxarea=cv2.contourArea(c)
                maxint = i
                i +=1
        box = cv2.approxPolyDP(contours[maxint], 15, True) #多边形拟合 True 代表封闭
        print(box.shape)
        poly = np.zeros(canny.shape)
        cv2.polylines(poly, [box],True, (255, 0, 0)) #连线
        cv2.imshow("2",poly)
        warped = four_point_transform(img, box)
        cv2.imshow("Warped", warped)

    else:
        ret,result = cv2.threshold(gray,127,255,cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        

    # 在原图上面或轮廓
    draw_img0 = cv2.drawContours(img,contours,-1,(0,255,255),3)
    if mode==1:
        cv2.imshow("canny",img)
    else:
        cv2.imshow("normal",draw_img0) 
        # 找出矩形对应的框
        bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
        # #在图片上画框
        for bbox in bounding_boxes:
            [x , y, w, h] = bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.waitKey()

start(MODE)