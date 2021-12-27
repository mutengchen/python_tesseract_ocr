import cv2
import os

img = cv2.imread('c3.png')
img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#二值化
ret,result = cv2.threshold(gray,127,255,cv2.THRESH_OTSU)
#将边缘增强过的图片的外部轮廓提取出来
contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
# 在原图上面或轮廓
draw_img0 = cv2.drawContours(img,contours,-1,(0,255,255),3)
# 找出矩形对应的框
# bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]


# # #在图片上画框
# for bbox in bounding_boxes:
#     print(bbox)
#     [x , y, w, h] = bbox
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)



# #显示图像
cv2.imshow("name",img)
# draw_img1 = cv2.drawContours(img.copy(),contours,1,(255,0,255),3)
# draw_img2 = cv2.drawContours(img.copy(),contours,2,(255,255,0),3)
# draw_img3 = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 3)
 
 
# print ("contours:类型：",type(contours))
# print ("第0 个contours:",type(contours[0]))
# print ("contours 数量：",len(contours))
# # print(contours)
# print(contours[1])
# print ("contours[0]点的个数：",len(contours[0]))
# print ("contours[1]点的个数：",len(contours[1])
# )
# 计算第一条轮廓的各阶矩,字典形式
# M = cv2.moments(contours[-1])  
# #获取中心点坐标二点位置

# center_x = int(M["m10"] / M["m00"])
# center_y = int(M["m01"] / M["m00"])
# print("center_x = %d center_y = %d",(center_x,center_y))

# cv2.imshow("img", img)
# cv2.imshow("draw_img1", draw_img1)
# cv2.imshow("draw_img2", draw_img2)
# cv2.imshow("draw_img3", draw_img3)


# for i in range(0,len(contours)): 
#     x, y, w, h = cv2.boundingRect(contours[i])  
#     cv2.rectangle(img, (x,y), (x+w,y+h), (153,153,0), 5)
#     new_image=img[y+2:y+h-2,x+2:x+w-2]    # 先用y确定高，再用x确定宽
#     cv2.imwrite( str(i)+".jpg",new_image)
#     print (i)
cv2.waitKey()