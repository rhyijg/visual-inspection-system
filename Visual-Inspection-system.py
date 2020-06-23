import numpy as np
import cv2
import matplotlib.pyplot as plt
def show(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()
image=cv2.imread("./010.png")
image1=image[200:2144,200:2792]
print("*******裁剪完成图片********")
show(image1)
# image1 = cv2.cvtColor(image1,cv2.COLOR_RGB2BGR)
# #image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
image1.shape
sp = image1.shape 
height = sp[0]  
width = sp[1] 
new = np.zeros((height,width,3), np.uint8)
print("*******将图片灰度化********")
for i in range(height):   
    for j in range(width):   
        new[i,j] = 0.3 * image1[i,j][0] + 0.59 * image1[i,j][1] + 0.11 * image1[i,j][2]   
img_gauss=cv2.GaussianBlur(image1,(7,7),0)
ret, binary = cv2.threshold(img_gauss,127,255, cv2.THRESH_BINARY_INV)
print("*******将图片二值化********")
show(binary)
# 闭运算
kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
print("*******创建二值化图像窗口********")
cv2.namedWindow("closing",cv2.WINDOW_NORMAL)
cv2.imshow("closing", closing)

closing = cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY)
shapes = {'triangle': 0, 'rectangle': 0, 'polygons': 0, 'circles': 0}
print("*******寻找轮廓********")
contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in range(len(contours)):  #遍历轮廓
    # 提取与绘制轮廓
    cv2.drawContours(image1,contours, c, (0, 255, 0), 2)
    # 轮廓逼近
    epsilon = 0.01 * cv2.arcLength(contours[c], True)
    approx = cv2.approxPolyDP(contours[c], epsilon, True)
    # 分析几何形状
    corners = len(approx)
    print(corners)
    shape_type = ""
    if corners == 3:
        count = shapes['triangle']
        count = count + 1
        shapes['triangle'] = count
        shape_type = "三角形"
    if corners == 4:
        count = shapes['rectangle']
        count = count + 1
        shapes['rectangle'] = count
        shape_type = "矩形"
        #寻找其最小外接矩形
        rect = cv2.minAreaRect(contours[c])
        #获取长宽
        aaa=rect[1][0]
        bbb=rect[1][1]

        k = 0
        if aaa < bbb:
            k = aaa
            aaa = bbb
            bbb = k
        length = aaa*0.477 # 实际长度
        width = bbb*0.477 # 实际宽度
        print("长： %.3f, 宽： %.3f， 形状: %s " % (length, width, shape_type))
    if corners >= 10:
        count = shapes['circles']
        count = count + 1
        shapes['circles'] = count
        shape_type = "圆形"
        # 计算直径
        x_circle, y_circle, w_pixel_circle, h_pixel_circle = cv2.boundingRect(contours[c])
        d_circle = w_pixel_circle*0.477
        print("直径： %.3f, 形状: %s " % (d_circle, shape_type))
    if 4 < corners < 10:
        count = shapes['polygons']
        count = count + 1
        shapes['polygons'] = count
        shape_type = "多边形"
    cv2.namedWindow("contour", 0)
    cv2.imshow("contour", image1)
print("*******完成计算*******")
cv2.waitKey(0)
cv2.destroyAllWindows()