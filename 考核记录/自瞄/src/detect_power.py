import cv2
import numpy as np

isactive = False

def preprocess(img):

    lower_pink = np.array([100, 100, 200])
    upper_pink = np.array([170, 255, 230])

    kernel = np.ones((3, 3), np.uint8)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(img_hsv, lower_pink, upper_pink)


    B, G, R = cv2.split(img)
    red_diff = cv2.subtract(R, ((G.astype(np.float32)+B.astype(np.float32))/2).astype(np.uint8))
    _, mask_diff = cv2.threshold(red_diff, 80, 255, cv2.THRESH_BINARY)


    mask = cv2.bitwise_and(mask_hsv, mask_diff)

    mask = cv2.dilate(mask, kernel=kernel, iterations=5)  # 膨胀，填补空洞
    mask = cv2.erode(mask, kernel=kernel, iterations=5)   # 腐蚀，去除噪点
    return mask

num = 0
center_x, center_y = -1, -1


def Get_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 查找轮廓,参数:图片,轮廓检索模式,轮廓近似方法
    global num
    global center_x, center_y
    num += 1
    if num == 1:
        minarea = 5000
        minid = -1
        for i, cnt in  enumerate(contours):
            area = cv2.contourArea(cnt)  # 计算轮廓面积
            if area > 50:  # 过滤掉小面积的轮廓
                if hierarchy[0][i][2] >= 0:continue
                if hierarchy[0][i][3] >= 0:continue
                if area < minarea:
                    minarea = area
                    minid = i
        if minid != -1:
            x, y, w, h = cv2.boundingRect(contours[minid])
            center_x = x + w // 2
            center_y = y + h // 2
            print("中心点坐标: ", center_x, center_y)
    imgContour = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    maxarea, maxid = -1, -1  # 创建一个空白图像用于绘制轮廓
    for i, cnt in  enumerate(contours):
        area = cv2.contourArea(cnt)  # 计算轮廓面积
        print(area)
        if area > 0:  # 过滤掉小面积的轮廓
            cv2.drawContours(imgContour, cnt, -1, (255, 255, 0), 5)  # 绘制轮廓,参数:图片,轮廓,-1表示绘制所有轮廓,(B,G,R),厚度
            if hierarchy[0][i][2] >= 0:continue
            if hierarchy[0][i][3] >= 0:continue
            if area > maxarea:
                maxarea = area
                maxid = i
    if maxid != -1:
        cv2.drawContours(imgContour, contours[maxid], -1, (0, 255, 255), 5)
        x, y, w, h = cv2.boundingRect(contours[maxid])
        cx, cy = x + w // 2, y + h // 2
        if cx > center_x and cy > center_y:
            cx, cy = int(center_x + (cx-center_x)*1.2), int(center_y + (cy - center_y)*1.4)
        elif cx < center_x and cy > center_y:
            cx, cy = int(center_x - (center_x-cx)*1.2), int(center_y + (cy - center_y)*1.4)
        elif cx < center_x and cy < center_y:
            cx, cy = int(center_x - (center_x-cx)*1.2), int(center_y - (center_y - cy)*1.4)
        elif cx > center_x and cy < center_y:
            cx, cy = int(center_x + (cx-center_x)*1.2), int(center_y - (center_y - cy)*1.4)
        cv2.circle(imgContour, (cx, cy), 7, (0, 0, 255), 2)
        if maxarea > 5500:
            global isactive
            isactive = True
        else:
            isactive = False
    return imgContour
            

def main():
    try:
        cap = cv2.VideoCapture("D:\\Users\\LD\\Desktop\\文件\\rm\\考核记录\\自瞄\\source\\power.mp4")#从文件中获取视频
    except Exception as e:
        print("⚠️ 无法打开视频文件: ", e)
        return
    print("视频是否打开: ", cap.isOpened())
    while True:
        success, img = cap.read()#从cap中读取帧,success:bool 表示是否读取到帧,img为读取到的帧
        if not success:   # 视频读到最后一帧 或 打不开
            print("⚠️ 无法读取帧（可能是视频结束或路径错误）")
            break
        img = cv2.resize(img, (540, 960))  # 修改图片大小,参数:图片,新大小(宽,高)(x,y)
        print(img.shape)
        mask = preprocess(img)
        img_cnt = Get_contours(mask)
         # 显示处理后的图像
        cv2.putText(img_cnt, "yellow:to be hitted", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(img_cnt, "bule:hitted", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        if isactive:
            cv2.putText(img_cnt, "Status: Active", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img_cnt, "Status: Inactive", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #合并img and img_cnt
        img = cv2.addWeighted(img, 0.5, img_cnt, 0.5, 0)
        cv2.imshow('video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):#按q退出窗口
            break
if __name__ == "__main__":
    main()