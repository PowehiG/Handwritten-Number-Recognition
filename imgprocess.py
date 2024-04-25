import cv2
import numpy as np


def image_preprocessing():
    # 读取图片
    img = cv2.imread("./example/8.jpg")

    # 图像转灰度
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 图像高斯滤波
    gauss_img = cv2.GaussianBlur(gray_img, (5, 5), 0, 0, cv2.BORDER_DEFAULT)

    # 边缘检测
    img_edge1 = cv2.Canny(gauss_img, 100, 200)
    #cv2.imshow("edge", img_edge1)
    # 获取原始图像的宽和高
    high = img.shape[0]  # 矩阵行——高
    width = img.shape[1]  # 矩阵列——宽

    # 初始化高宽和
    add_width = np.zeros(high, dtype=int)
    add_high = np.zeros(width, dtype=int)

    # 计算每一行的值的和
    for h in range(high):
        for w in range(width):
            add_width[h] = add_width[h] + img_edge1[h][w]

    # 计算每一列的值的和
    for w in range(width):
        for h in range(high):
            add_high[w] = add_high[w] + img_edge1[h][w]

    # 初始化上下边界为宽度总值最大的值的索引
    account_high_up = np.argmax(add_width)
    account_high_down = np.argmax(add_width)

    # 将上边界坐标值上移，直到没有遇到白色点停止，此为数字的上边界
    while add_width[account_high_up] != 0:
        account_high_up = account_high_up - 1

    # 将下边界坐标值下移，直到没有遇到白色点停止，此为数字的下边界
    while add_width[account_high_down] != 0:
        account_high_down = account_high_down + 1

    # 初始化左右边界为宽度总值最大的值的索引
    account_width_left = np.argmax(add_high)
    account_width_right = np.argmax(add_high)

    # 将左边界坐标值左移，直到没有遇到白色点停止，此为数字的左边界
    while add_high[account_width_left] != 0:
        account_width_left = account_width_left - 1

    # 将右边界坐标值右移，直到没有遇到白色点停止，此为数字的右边界
    while add_high[account_width_right] != 0:
        account_width_right = account_width_right + 1

    #print(account_high_down,account_high_up,account_width_left,account_width_right)
    # 求出宽和高的间距
    width_spacing = account_width_right - account_width_left
    high_spacing = account_high_down - account_high_up

    # 求出宽和高的间距差
    poor = width_spacing - high_spacing
    #print(poor)
    # 将数字进行正方形分割，目的是方便之后进行图像压缩
    if poor > 0:
        tailor_image = img[account_high_up  :account_high_down ,
                       account_width_left:account_width_right]

    else:

        tailor_image = img[account_high_up :account_high_down,
                       account_width_left :account_width_right ]

    # ==================================================== #
    # ======================小图处理======================= #
    #cv2.imshow("cut", img)
    # 将裁剪后的图片进行灰度化
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯去噪
    gauss_img = cv2.GaussianBlur(gray_img, (5, 5), 0, 0, cv2.BORDER_DEFAULT)

    # 将图像形状调整到28*28大小
    zoom_image = cv2.resize(gauss_img, (28, 28))
    # 获取图像的高和宽
    high = zoom_image.shape[0]
    wide = zoom_image.shape[1]

    # 将图像每个点的灰度值进行阈值比较
    for h in range(high):
        for w in range(wide):

            # 若灰度值大于100，则判断为背景并赋值0，否则将深灰度值变白处理
            if zoom_image[h][w] > 100:
                zoom_image[h][w] = 0
            else:
                zoom_image[h][w] = 255 - zoom_image[h][w]

    # ==================================================== #
    cv2.imshow("final", zoom_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return zoom_image
