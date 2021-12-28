import cv2
import os
import numpy as np

from ocr import OcrSdk

MODE = 1 #0表示普通识别，1表示算法优化过的
#使用canny对边缘进行加强


def order_points(pts):
   # 一共4个坐标点
    rect = np.zeros((4, 2), dtype = "float32")
 
    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # 计算右上和左下
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


#仿射变换，把倾斜的图片转换成水平
def four_point_transform(image, pts):
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    print(rect)
    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    print("------------")
    print(widthA)
    print(widthB)
    maxWidth = max(int(widthA), int(widthB))
    print(maxWidth)
    
    print("------------")
 
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    print("------------")
    print(heightA)
    print(heightB)
    maxHeight = max(int(heightA), int(heightB))
    print(maxHeight)
    print("------------")
 
    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    print("矩阵dst：")
    print(dst)
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    print(M)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # cv2.imshow('wrapper',warped)
    # 返回变换后结果
    return warped



def start(mode):
    img = cv2.imread('crop_1.png')
    img = cv2.resize(img,(0,0),fx=0.6,fy=0.6)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode==1 :
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        canny=cv2.Canny(dilate,30,120,3)
        #将边缘增强过的图片的外部轮廓提取出来
        contours, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        docCnt = None
        if len(contours) > 0:
        # 按轮廓大小降序排列
            cnts = sorted(contours, key=cv2.contourArea, reverse=True)
            print("c len:"+str(len(cnts)))
            for c in cnts:
                # 近似轮廓
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                # 如果我们的近似轮廓有四个点，则确定找到了纸
                if len(approx) == 4:
                    docCnt = approx
                    break
        # 对原始图像应用四点透视变换，以获得纸张的俯视图
        paper = four_point_transform(img, docCnt.reshape(4, 2))
        #调用tessract进行文字识别
        OcrSdk().start(paper)
    else:
        ret,result = cv2.threshold(gray,127,255,cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        

    # 在原图上面或轮廓
    # draw_img0 = cv2.drawContours(img,contours,-1,(0,255,255),3)
    # if mode==1:
    #     cv2.imshow("canny",img)
    # else:
    #     cv2.imshow("normal",draw_img0) 
    #     # 找出矩形对应的框
    #     bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
        
    #     # #在图片上画框
    #     for bbox in bounding_boxes:
    #         [x , y, w, h] = bbox
    #         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.waitKey()

start(MODE)