import cv2
import numpy as np

# 世界坐标 (一个 14cm x 12cm 的装甲板板)
w, h = 0.14, 0.12
objectPoints = np.array([
    [-w/2, -h/2, 0],
    [ w/2, -h/2, 0],
    [ w/2,  h/2, 0],
    [-w/2,  h/2, 0]
], dtype=np.float32)

# 相机参数
fx, fy, cx, cy = 1000, 1000, 500, 350
cameraMatrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float32)

distCoeffs = np.zeros((5,1))  # 假设无畸变


def transform_to_base(obj_in_cam):
    """
    把 OpenCV 默认相机坐标系 (右x, 下y, 前z)
    转换为 基坐标系 (前x, 左y, 上z)
    """
    T = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ], dtype=np.float32)

    return T @ obj_in_cam.reshape(3)

def sort_corners(approx):
    """
    输入:
        approx: 形状 (4,1,2) 或 (4,2) 的矩形四个点
    输出:
        imagePoints: 按 [左下, 右下, 右上, 左上] 顺序排好的 (4,2) ndarray
    """

    pts = approx.reshape(4, 2)  # 转成 (4,2)
    # 计算中心点
    cx, cy = np.mean(pts, axis=0)

    sorted_pts = np.zeros((4, 2), dtype=np.float32)

    for p in pts:
        x, y = p
        if x < cx and y > cy:
            sorted_pts[0] = p  # 左下
        elif x > cx and y > cy:
            sorted_pts[1] = p  # 右下
        elif x > cx and y < cy:
            sorted_pts[2] = p  # 右上
        elif x < cx and y < cy:
            sorted_pts[3] = p  # 左上

    return sorted_pts, cx, cy


def preprocess_image(image):

    lower_green = np.array([35,  43,  46])   # 下界 (H, S, V)
    upper_green = np.array([85, 255, 255])   # 上界 (H, S, V)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 转换到HSV空间
    mask = cv2.inRange(hsv_img, lower_green, upper_green)  # 创建掩码
    return mask

def find_counters(mask,img):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)  # 计算轮廓面积
        print(area)
        if area > 300:  # 过滤掉小面积的轮廓
            peri = cv2.arcLength(cnt, True)  # 计算轮廓周长,参数:轮廓,是否闭合
            print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # 多边形逼近,参数:轮廓,精度(越小越精确),是否闭合
            objCor = len(approx)  # 获取多边形的顶点数

            y_offset = 60
            if objCor == 4 :

                sorted_corners, cx, cy = sort_corners(approx)
                cx, cy = int(cx), int(cy)
                imagePoints = np.array(sorted_corners, dtype=np.float32)
                _, _, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)

                obj_in_cam = tvec.reshape(3)
                obj_in_base = transform_to_base(obj_in_cam)

                for i in range(len(approx)):
                    cv2.line(img, tuple(approx[i][0]), tuple(approx[(i + 1) % len(approx)][0]), (0, 255, 255), 2)  # 画线,参数:图片,起点,终点,颜色,粗细
                cv2.putText(img, f"board[{num + 1}]", (15, 60 + num * y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(img, f"x[{obj_in_base[0]:.2f}m];y[{obj_in_base[1]:.2f}m];z[{obj_in_base[2]:.2f}m]", (15, 60 + num * y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.circle(img, (cx, cy), 3, (0, 255, 255), cv2.FILLED)  # 画中心点,参数:图片,中心点,半径,颜色,填充
                num += 1
    return img


    
def main():
    try:
        cap = cv2.VideoCapture("D:\\Users\\LD\\Desktop\\文件\\rm\\考核记录\\自瞄\\source\\target.mp4")#从文件中获取视频
    except Exception as e:
        print("⚠️ 无法打开视频文件: ", e)
        return
    print("视频是否打开: ", cap.isOpened())
    while True:
        success, img = cap.read()#从cap中读取帧,success:bool 表示是否读取到帧,img为读取到的帧
        if not success:   # 视频读到最后一帧 或 打不开
            print("⚠️ 无法读取帧（可能是视频结束或路径错误）")
            break
        mask = preprocess_image(img)
        img = find_counters(mask,img)
         # 显示处理后的图像
        cv2.imshow('video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):#按q退出窗口
            break
if __name__ == "__main__":
    main()