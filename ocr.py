import cv2
import pytesseract

import datetime
import time

# 优化思路:
# 1.图片高斯模糊去噪点
# 2.转换成灰度图，增强文字轮廓
# 3.切割关键部分，提高识别效率，和识别率
# def chuli(image):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # 大津法二值化
#     # retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
#     # # 腐蚀和膨胀是对白色部分而言的，膨胀，白区域变大，最后的参数为迭代次数
#     # dst = cv2.dilate(dst, None, iterations=1)
#     # # 腐蚀，白区域变小
#     # dst = cv2.erode(dst, None, iterations=2)
#     # return dst
#     return gray

def boxDraw(img,boxes):
    temp = boxes.split('\n')
    for index,temp1 in enumerate(temp):
        temp2 = temp1.split()
        if index==0:
            continue
        if len(temp1)>1:
            cv2.rectangle(img,(int(temp2[1]),int(temp2[2])),(int(temp2[3]),int(temp2[4])),(0,0,255))

def dataDraw(img,data):
    temp = data.split('\n')
    print(type(img))
    t = []
    max_width = 0
    max_height = 0
    for index,temp1 in enumerate(temp):
        temp2 = temp1.split()
        if index==0:
            continue
        #过滤掉单个字符的文本框，只找类似区域的文本框
        if len(temp1)>1 and int(temp2[8])>100:
            x1 = int(temp2[6])-10
            y1 = int(temp2[7])-10
            x2 = int(temp2[6])+int(temp2[8])+10
            y2 = int(temp2[7])+int(temp2[9])+10
            #在图上画框
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255))
            #把对应的图片都给扣下来
            imgcrop = img[y1:y2,x1:x2]
            print(imgcrop.shape)
            t.append(imgcrop)
            filename = "result_"+str(index)+".png"
    print(len(t))
    result = cv2.vconcat(t)
    cv2.imshow("result",result)

class OcrSdk:
    def start(self,img):
      
        custom_config = r'-l chi_sim eng equ --oem 3 --psm 7' 
        result = pytesseract.image_to_string(img, config=custom_config)
        boxes  = pytesseract.image_to_boxes(img)
        data = pytesseract.image_to_data(img)
        print(result)
        dataDraw(img,data)
        cv2.imshow('ocr_result',img)
        cv2.waitKey()

obj = OcrSdk()
filename = '1111.png'
img = cv2.imread(filename)
img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
obj.start(img)











