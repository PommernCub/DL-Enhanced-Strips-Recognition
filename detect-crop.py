
import os
import cv2
import numpy as np
import csv

from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.ndimage import gaussian_filter
from scipy.integrate import trapz, simps

import matplotlib.pyplot as plt

"""
使用 OpenCV 方法对试纸条颜色进行判断,数据来源为YOLO标记或输出的裁剪坐标
1. 对裁剪数据进行区域检测 (T/C线)
2. 灰度处理
3. 进行灰度/RGB/HSV色度识别
4. 输出 (图片分析和色度数据) 结果文件

"""


def convert(img_size,x,y,w,h):

    x = x*img_size[1] # img_size(y,x,c)
    w = w*img_size[1]
    y = y*img_size[0]
    h = h*img_size[0]
    
    x1 = int((x+1) - w/2.0)  # opencv裁剪需要整数像素值
    y1 = int((y+1) - h/2.0)
    x2 = int((x+1) + w/2.0)
    y2 = int((y+1) + h/2.0)

    return (x1,y1,x2,y2)

               
def img_crop(img,lab_path):

    # 读取标签文件
    data = []
    with open(lab_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            data.append(line.split())
    
    # 提前按照大致区域为T/C赋值，避免未检测出的标签导致报错
    y1,y2,x1,x2 = int(0.24*img.shape[0]),int(0.3*img.shape[0]),int(0.2*img.shape[1]),int(0.8*img.shape[1])
    imgC = img[y1:y2, x1:x2]
    contourC = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]], dtype=np.int32)
    y1,y2,x1,x2 = int(0.62*img.shape[0]),int(0.68*img.shape[0]),int(0.2*img.shape[1]),int(0.8*img.shape[1])
    imgT = img[y1:y2, x1:x2]
    contourT = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]], dtype=np.int32)
        
    for lab in data:
        classes,cx,cy,w,h = int(lab[0]),float(lab[1]),float(lab[2]),float(lab[3]),float(lab[4])        

        x1,y1,x2,y2 = convert(img.shape,cx,cy,w,h)  # yolo标签格式转换为框绝对像素坐标
    
        # 按照框坐标裁剪图片
        if classes == 1:
            imgC = img[y1:y2, x1:x2]
            contourC = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]], dtype=np.int32)
        elif classes == 2:
            imgT = img[y1:y2, x1:x2]
            contourT = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]], dtype=np.int32)
    
    return (imgT,contourT,imgC,contourC)


def img_contour1(img):
    
    # 转换为灰度图像
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊降低噪声
    blurred = cv2.GaussianBlur(gray_image, (11, 11), 0)

    # 计算光照梯度
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    gradient_magnitude = (255 * gradient_magnitude / gradient_magnitude.max()).astype(np.uint8)

    # 应用阈值
    _, thresholded = cv2.threshold(gradient_magnitude, 30, 255, cv2.THRESH_BINARY)
    
    # 获取轮廓
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 过滤轮廓并获取较大的区域
    filtered_contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    # 绘制轮廓
    result = image.copy()
    contour_img = cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)
    
    return (contour_img)

def img_contour2(img):
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义粉红色阈值范围（需要根据实际图像进行调整）
    lower_pink = np.array([100, 50, 100])
    upper_pink = np.array([180, 255, 255])

    # 创建蒙版并进行颜色筛选
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # 去除噪点
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选出长方形
    rects = []
    for contour in contours:
        # 过滤掉较小的轮廓
        if cv2.contourArea(contour) < 500:
            continue

    # 计算轮廓的外接矩形
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 检查形状是否为长方形
    width = int(rect[1][0])
    height = int(rect[1][1])
    aspect_ratio = float(width) / height
    if 0.8 <= aspect_ratio <= 1.2:  # 长宽比在这个范围内认为是长方形
        rects.append(box)

    # 在原图上绘制长方形轮廓
    for rect in rects:
        contour_img = cv2.drawContours(image, [rect], 0, (0, 255, 0), 2)
    
    return (contour_img)


def extract_rgb(imgT,imgC,save_path):
    """
    传入待分析的图片，提取图片的rgb颜色分布
    返回rgb峰值和面积

    Parameters
    ----------
    img : numpy array (使用 cv2.imread函数打开的图片)
        待处理分析的图片数组，opencv格式打开.
    save_path : string
        传入图像的rgb分析结果图片（如有）的储存路径.        

    Returns
    -------
    rgb_hist: list
        用于储存rgb分布峰值及其积分面积的列表, rgb[0][1][2]分别为red,green,blue通道峰值大小，
        rgb[3][4][5]分别为red,green,blue通道面积大小.

    """
    
    bgr_hist1 = [] # [0]:blue_peak, [1]:blue_area, ..., [4]:red_peak, [5]:red_area
    bgr_hist2 = []

    img1 = imgT.copy()
    img2 = imgC.copy()
    # BGR->RGB
    img1[:,:,2], img1[:,:,0] = imgT[:,:,0], imgT[:,:,2]
    img2[:,:,2], img2[:,:,0] = imgC[:,:,0], imgC[:,:,2]
    
    # 绘图 
    # plt.clf() # 绘图前清空图片
    # plt.subplot(221), plt.imshow(img1), plt.title("Image_T")
    # plt.subplot(223), plt.imshow(img2), plt.title("Image_C")
    color = ('b','g','r')
    for i,col in enumerate(color):
        # 计算img_T图像rgb色彩分布的直方图
        histr1 = cv2.calcHist([imgT],[i],None,[256],[0,256]) # i = 0:blue, 1:green, 2:red        
        histr1 = histr1/histr1.sum()
        """
        # plt.subplot(222), plt.plot(histr1,color=col)#, plt.title("Histogram_T")
        plt.subplot(222), plt.plot(histr1,color=col), plt.title("RGB histogram")
        """
        max_indx = np.argmax(histr1) # max value index
        # 储存该通道峰对应值
        bgr_hist1.append(max_indx)
        """
        plt.plot(max_indx,histr1[max_indx],col+'-o')
        show_max = '(' + str(max_indx) + ', ' + str(round(histr1[max_indx][0],3)) + ')'
        plt.annotate(show_max,xytext=(max_indx,histr1[max_indx]),xy=(max_indx,histr1[max_indx]),color=col)
        """
        # 计算峰的面积 (峰值*color_index)
        bgr_hist1.append((histr1.T*np.arange(256)).sum())
        
        # 计算img_C图像rgb色彩分布的直方图
        histr2 = cv2.calcHist([imgC],[i],None,[256],[0,256])
        histr2 = histr2/histr2.sum()
        """
        # plt.subplot(224), plt.plot(histr2,color=col)#, plt.title("Histogram_C")        
        histx = np.linspace(0, len(histr2)-1, len(histr2))  # 绘制条形图bar需要的横坐标范围
        plt.subplot(222), plt.bar(histx,histr2[:,0], width=0.8, color=col, alpha=0.5)
        """
        max_indx = np.argmax(histr2) # max value index
        bgr_hist2.append(max_indx)
        """
        plt.plot(max_indx,histr2[max_indx],col+'-o')
        show_max = '(' + str(max_indx) + ', ' + str(round(histr2[max_indx][0],3)) + ')'
        plt.annotate(show_max,xytext=(max_indx,histr2[max_indx]),xy=(max_indx,histr2[max_indx]),color=col)       
        """
        # 计算峰的面积 (峰值*color_index)
        bgr_hist2.append((histr2.T*np.arange(256)).sum())

    """
    plt.xlim([0,256])
    # plt.show()
    # plt.savefig(save_path + '.png')
    """
    
    # 计算rgb峰和面积比值
    rgb_peak,rgb_area = calc_ratio(bgr_hist1,bgr_hist2,'rgb')
    # 输出列表内容（hsv通道T-red平均值,C-r平均值，T-green平均值,C-g平均值，T-blue平均值,C-b平均值，T/C比值）
    rgb_output = [bgr_hist1[5],bgr_hist2[5],bgr_hist1[3],bgr_hist2[3],bgr_hist1[1],bgr_hist2[1],rgb_area]
    
    return (rgb_peak, rgb_area, rgb_output)


def extract_hsv(imgT,imgC,save_path):    
    """
    传入待分析的图片，提取图片的hsv颜色分布
    返回hsv峰值和面积

    Parameters
    ----------
    img : numpy array (使用 cv2.imread函数打开的图片)
        待处理分析的图片数组，opencv格式打开.
    save_path : string
        传入图像的分析结果图片（如有）的储存路径.        

    Returns
    -------
    hsv_hist: list
        用于储存hsv分布峰值及其积分面积的列表, hsv[0][1][2]分别为色相hue,饱和度saturation,明度value通道峰值大小，
        rgb[3][4][5]分别为hue,saturation,value通道面积大小.

    """
    
    # [0]:hue_peak, [1]:hue_area, [2]:saturate_peak, [3]:saturation_area, [4]:value_peak, [5]:value_area
    hsv_hist1 = [] 
    hsv_hist2 = []

    img1 = imgT.copy()
    img2 = imgC.copy()
    # BGR->RGB
    img1[:,:,2], img1[:,:,0] = imgT[:,:,0], imgT[:,:,2]
    img2[:,:,2], img2[:,:,0] = imgC[:,:,0], imgC[:,:,2]
    # 转换为HSV空间
    imgT = cv2.cvtColor(imgT, cv2.COLOR_BGR2HSV)
    imgC = cv2.cvtColor(imgC, cv2.COLOR_BGR2HSV)
    
    # 绘图 
    # plt.clf() # 绘图前清空图片
    # plt.subplot(221), plt.imshow(img1), plt.title("Image_T")
    # plt.subplot(223), plt.imshow(img2), plt.title("Image_C")
    color = ('m','k','orange')
    for i,col in enumerate(color):
        # 计算img_T图像hsv色彩分布的直方图
        histr1 = cv2.calcHist([imgT],[i],None,[256],[0,256]) # i = 0:hue色相, 1:saturation饱和度, 2:value明度
        histr1 = histr1/histr1.sum()
        """
        # plt.subplot(222), plt.plot(histr1,color=col)#, plt.title("Histogram_T")
        plt.subplot(224), plt.plot(histr1,color=col), plt.title("HSV histogram")
        """
        max_indx = np.argmax(histr1) # max value index
        # 储存该通道峰对应值
        hsv_hist1.append(max_indx)
        """
        plt.plot(max_indx,histr1[max_indx],'-o',color=col)
        show_max = '(' + str(max_indx) + ', ' + str(round(histr1[max_indx][0],3)) + ')'
        plt.annotate(show_max,xytext=(max_indx,histr1[max_indx]),xy=(max_indx,histr1[max_indx]),color=col)
        """
        # 计算峰的面积（峰值*color_index）        
        hsv_hist1.append((histr1.T*np.arange(256)).sum())
        
        # 计算img_C图像hsv色彩分布的直方图
        histr2 = cv2.calcHist([imgC],[i],None,[256],[0,256])
        histr2 = histr2/histr2.sum()
        """
        histx = np.linspace(0, len(histr2)-1, len(histr2))  # 绘制条形图bar需要的横坐标范围
        plt.subplot(224), plt.bar(histx,histr2[:,0], width=0.9, color=col, alpha=0.6)
        # plt.subplot(224), plt.plot(histr2,color=col)#, plt.title("Histogram_C")        
        """
        max_indx = np.argmax(histr2) # max value index
        hsv_hist2.append(max_indx)
        """
        plt.plot(max_indx,histr2[max_indx],'-o',color=col)
        show_max = '(' + str(max_indx) + ', ' + str(round(histr2[max_indx][0],3)) + ')'
        plt.annotate(show_max,xytext=(max_indx,histr2[max_indx]),xy=(max_indx,histr2[max_indx]),color=col)       
        """
        # 计算峰的面积（峰值*color_index）        
        hsv_hist2.append((histr2.T*np.arange(256)).sum())
    
    """
    plt.xlim([0,256])
    # plt.show()
    # plt.savefig(save_path + '.png')
    """   
    
    # 计算hsv峰和面积比值
    hsv_peak,hsv_area = calc_ratio(hsv_hist1,hsv_hist2,'hsv')
    # 输出列表内容（hsv通道T-h色相平均值,C-h平均值，T-s饱和度平均值,C-s平均值，T-v明度平均值,C-v平均值，T/C比值）
    hsv_output = [hsv_hist1[1],hsv_hist2[1],hsv_hist1[3],hsv_hist2[3],hsv_hist1[5],hsv_hist2[5],hsv_area]
    
    return (hsv_peak, hsv_area, hsv_output)


def extract_gray(imgT,imgC,save_path):
    """
    传入待分析的图片，提取图片的灰度图颜色分布
    返回灰度峰值和面积

    Parameters
    ----------
    img : numpy array (使用 cv2.imread函数打开的图片)
        待处理分析的图片数组，opencv格式打开.
    save_path : string
        传入图像的分析结果图片（如有）的储存路径.        

    Returns
    -------
    gray_hist: list
        用于储存灰度分布峰值及其积分面积的列表，
        gray[0][1]分别为通道峰值和面积大小.

    """
    
    # [0]:gray_peak, [1]:gray_area
    gray_hist1 = [] 
    gray_hist2 = []
    # 转换为灰度空间
    imgT = cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)
    imgC = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)
    
    # 绘图 
    # plt.clf() # 绘图前清空图片
    # plt.subplot(221), plt.imshow(imgT,cmap='gray'), plt.title("Image_T")
    # plt.subplot(223), plt.imshow(imgC,cmap='gray'), plt.title("Image_C")
    
    # 计算img_T图像灰度分布的直方图
    histr1 = cv2.calcHist([imgT],[0],None,[256],[0,256])
    histr1 = histr1/histr1.sum()
    """
    # plt.subplot(222), plt.plot(histr1,color='k')#, plt.title("Histogram_T")
    plt.subplot(223), plt.plot(histr1,color='k'), plt.title("Gray histogram")
    """
    max_indx = np.argmax(histr1) # max value index    
    # 储存该通道峰对应值
    gray_hist1.append(max_indx)
    """
    plt.plot(max_indx,histr1[max_indx],'k-o')
    show_max = '(' + str(max_indx) + ', ' + str(round(histr1[max_indx][0],3)) + ')'
    plt.annotate(show_max,xytext=(max_indx,histr1[max_indx]),xy=(max_indx,histr1[max_indx]),color='k')
    """
    # 计算峰的面积（峰值*color_index）        
    gray_hist1.append((histr1.T*np.arange(256)).sum())
    
    # 计算img_C图像灰度分布的直方图
    histr2 = cv2.calcHist([imgC],[0],None,[256],[0,256])
    histr2 = histr2/histr2.sum()
    """
    # plt.subplot(224), plt.plot(histr2,color='k')#, plt.title("Histogram_C")
    histx = np.linspace(0, len(histr2)-1, len(histr2))  # 绘制条形图bar需要的横坐标范围
    plt.subplot(223), plt.bar(histx,histr2[:,0], width=0.8, color='k', alpha=0.8)       
    """
    max_indx = np.argmax(histr2) # max value index
    gray_hist2.append(max_indx)
    """      
    plt.plot(max_indx,histr2[max_indx],'k-o')
    show_max = '(' + str(max_indx) + ', ' + str(round(histr2[max_indx][0],3)) + ')'
    plt.annotate(show_max,xytext=(max_indx,histr2[max_indx]),xy=(max_indx,histr2[max_indx]),color='k')
    """
    # 计算峰的面积（峰值*index）        
    gray_hist2.append((histr2.T*np.arange(256)).sum())

    """
    plt.xlim([0,256])
    # plt.show()
    # plt.savefig(save_path + '.png')
    """
    
    # 计算灰度通道峰和面积比值
    gray_peak,gray_area = calc_ratio(gray_hist1,gray_hist2,'gray')
    # 输出列表内容（灰度通道T平均值,C平均值，T/C比值）
    gray_output = [gray_hist1[1],gray_hist2[1],gray_area]
    
    return (gray_peak, gray_area, gray_output)


def extract_intensity(image,mode,save_path):
    
    if mode == 'Red':
        img = image[:,:,2]
    if mode == 'Green':
        img = image[:,:,1]
    if mode == 'Blue':
        img = image[:,:,0]
        
    if mode == 'Saturation':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img = 255-img[:,:,1]
    if mode == 'Value':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img = img[:,:,2]                
        
    if mode == 'Gray':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)            

    if (image.shape[0] >= image.shape[1]):
        y1,y2,x1,x2 = int(0.01*img.shape[0]),int(0.99*img.shape[0]),int(0.01*img.shape[1]),int(0.99*img.shape[1])
        img = img[y1:y2, x1:x2]
        imgL,imgW = img.shape[0],img.shape[1]
        intensity_profile = 255-np.mean(img, axis=1) # 纵向摆放的试纸条            
    else:
        y1,y2,x1,x2 = int(0.01*img.shape[0]),int(0.99*img.shape[0]),int(0.01*img.shape[1]),int(0.99*img.shape[1])
        img = img[y1:y2, x1:x2]
        imgL,imgW = img.shape[1],img.shape[0]
        intensity_profile = 255-np.mean(img, axis=0) # 横向摆放的试纸条


    # 使用高斯滤波器平滑数据        
    intensity_profile = gaussian_filter(intensity_profile, sigma=1) # sigma越大曲线越平滑，不然取不到准确峰信息
    
    # 找出峰的位置
    peaks, heights = find_peaks(intensity_profile, prominence=0.8)
    # peak不为两个时额外处理
    if (len(peaks) != 2):
        """
        print (name+':\t'+str(peaks))
        plt.imshow(img.T, extent=[0,imgL,70,imgW+70])
        plt.plot(intensity_profile, label=name, color='#ed5a65')
        plt.plot(peaks, intensity_profile[peaks], "x", label='Peaks')
        plt.title(name+' Intensity Profile - '+mode)            
        plt.show()
        """
        # 裁剪一下图片重新计算
        if (image.shape[0] >= image.shape[1]):
            y1,y2,x1,x2 = int(0.05*img.shape[0]),int(0.95*img.shape[0]),int(0.3*img.shape[1]),int(0.7*img.shape[1])
            img = img[y1:y2, x1:x2]
            intensity_profile = 255-np.mean(img, axis=1) # 纵向摆放的试纸条
            imgL,imgW = img.shape[0],img.shape[1]
        else:
            y1,y2,x1,x2 = int(0.3*img.shape[0]),int(0.7*img.shape[0]),int(0.05*img.shape[1]),int(0.95*img.shape[1])
            img = img[y1:y2, x1:x2]
            intensity_profile = 255-np.mean(img, axis=0) # 横向摆放的试纸条
            imgL,imgW = img.shape[1],img.shape[0]
        intensity_profile = gaussian_filter(intensity_profile, sigma=1)
        peaks, heights = find_peaks(intensity_profile, prominence=0.8)
        if (len(peaks) > 2):                
            # 去掉C线前面的峰
            while (len(heights['prominences']) > 2 and heights['prominences'][0] < 25):
                peaks = peaks[1:]
                heights['prominences'] = heights['prominences'][1:]            
            if (len(peaks) > 2):                
                # print (name+':\t'+str(peaks))
                # plt.imshow(img.T, extent=[0,imgL,70,imgW+70])
                # plt.plot(intensity_profile, label=name, color='#ed5a65')
                # plt.plot(peaks, intensity_profile[peaks], "x", label='Peaks')
                # plt.title(name+' Intensity Profile - '+mode)            
                # plt.show()
                #///////
                # # 去掉C线后面多余的峰，手动决定去掉哪个峰                
                # selected_peaks = [peaks[0]]  # 创建一个列表以保存用户选择的元素（里面有第一个峰的索引0）
                # # 输入想要保留的C峰的索引
                # for i in range(1):
                #     while True:
                #         try:
                #             index = int(input(f"图中从左到右，哪个峰是T线的峰（1 到 {len(peaks)}）: "))-1
                #             if 0 <= index < len(peaks):
                #                 selected_peaks.append(peaks[index])
                #                 break  # 有效索引，跳出循环
                #             else:
                #                 print("输入的索引超出范围，请重新输入。")
                #         except ValueError:
                #             print("输入无效，请输入一个整数索引。")
                # peaks = selected_peaks
                #///////
                # 不想手动，自动选取最高的两个峰
                peak_index = heights['prominences'].argsort()[-2:][::-1]  # 按峰高降序排列
                peaks = peaks[peak_index]
                heights['prominences'] = heights['prominences'][:2]
        """            
        # 重新作图
        plt.imshow(img.T, extent=[0,imgL,70,imgW+70])
        plt.plot(intensity_profile, label=name, color='#ed5a65')
        plt.plot(peaks, intensity_profile[peaks], "x", label='Peaks')
        plt.title(name+' Intensity Profile - '+mode)            
        plt.show()
        """
        
    # 由峰的位置来计算宽度和边界
    rel_h = [0.96, 0.9]  # 两个峰取不同的高度
    peak_properties = {}
    for i, peak in enumerate(peaks):
        width_props = peak_widths(intensity_profile, [peak], rel_height=rel_h[i])
        # 存储每个峰的结果
        peak_properties[peak] = {
            'width': width_props[0],
            'heights': width_props[1],
            'left_ips': width_props[2],
            'right_ips': width_props[3]
        }            
    # 矫正C峰基线过低的情况(C线基线应该较对称)
    baseLr = peak_properties[peaks[0]]['right_ips'][0]-peaks[0]
    baseLl = peaks[0]-peak_properties[peaks[0]]['left_ips'][0]
    width_half = peak_widths(intensity_profile, [peaks[0]], rel_height=0.5)
    if (abs(baseLr-baseLl) > width_half[0][0] or (baseLr+baseLl) > 2.0*width_half[0][0]):
        width_props = peak_widths(intensity_profile, [peaks[0]], rel_height=0.9)
        peak_properties[peaks[0]] = {
            'width': width_props[0],
            'heights': width_props[1],
            'left_ips': width_props[2],
            'right_ips': width_props[3]
        }            
    # 修改C峰基线过长的取值范围(C线位置不可能大于图像的一半)
    if peak_properties[peaks[0]]['right_ips'][0] > 0.5*imgL: #peak_properties[peaks[1]]['left_ips'][0]:
        left_bases = peak_properties[peaks[0]]['left_ips'][0]
        right_bases = 2*peaks[0]-left_bases
        peak_properties[peaks[0]]['right_ips'][0] = right_bases
        peak_properties[peaks[0]]['width'][0] = right_bases-left_bases
    # 修改T峰基线过长的取值范围(T线基线长度不应大于半峰宽的2.5倍，且应该比较对称)
    if (len(peaks) > 1):            
        baseLr = peak_properties[peaks[1]]['right_ips'][0]-peaks[1]
        baseLl = peaks[1]-peak_properties[peaks[1]]['left_ips'][0]
        width_half = peak_widths(intensity_profile, [peaks[1]], rel_height=0.5)
        if (abs(baseLr-baseLl) > width_half[0][0] or (baseLr+baseLl) > 2.5*width_half[0][0]):
            left_bases = peaks[1]-min(baseLr,baseLl)
            right_bases = peaks[1]+min(baseLr,baseLl)
            peak_properties[peaks[1]]['left_ips'][0] = left_bases
            peak_properties[peaks[1]]['right_ips'][0] = right_bases
            peak_properties[peaks[1]]['width'][0] = right_bases-left_bases

    # 绘制强度剖面
    if (mode == 'Gray'):
        plt.plot(intensity_profile, label=name)
        plt.title(name+' Intensity Profile - '+mode)
        plt.xlabel('Pixel')
        plt.ylabel('Intensity')
        # 在峰值位置绘制垂直线
        plt.plot(peaks, intensity_profile[peaks], "x", label='Peaks')
        # 在每个峰的宽度位置绘制水平线
        for i, peak in enumerate(peaks):            
            plt.hlines(peak_properties[peak]['heights'], peak_properties[peak]['left_ips'], peak_properties[peak]['right_ips'], color="C2")
        plt.show()
    
    # 积分计算峰的面积
    areas = [0,1]
    baselines = []
    for i, peak in enumerate(peaks):
        peak_left = round(peak_properties[peak]['left_ips'][0])
        peak_right = round(peak_properties[peak]['right_ips'][0])
        
        # 基线矫正
        baselines.append(peak_properties[peak]['heights'][0])
        
        # 定义峰的范围
        peak_data = intensity_profile[peak_left:peak_right]
        # peak_x = np.arange(len(intensity_profile))[peak_left:peak_right]
        
        # 计算峰的面积
        area = trapz(peak_data-baselines[i])  # 或者使用精度更高的 area = simps(peak_data, peak_x)
        areas[i] = area
    intensity_ratio = areas[1] / areas[0]
    
    if intensity_ratio > 1.0:
        intensity = [areas[0], areas[1], areas[0] / areas[1]]
    else:    
        intensity = [areas[1], areas[0], intensity_ratio]
    
    return (intensity)
        
    


def calc_ratio(histT,histC,mode):
    
    # RGB模式：ratio = T(blue+red)/C(blue+red)
    if mode == 'rgb':
        peakT = histT[0] + histT[4]
        peakC = histC[0] + histC[4]
        if peakC <= 0.001:
            peakC = 0.001
        areaT = histT[1] + histT[5]
        areaC = histC[1] + histC[5]
        if areaC <= 0.001:
            areaC = 0.001
        ratio = [peakT/peakC, areaT/areaC]
    
    # HSV模式：ratio = T(value)/C(value)-T(saturation)/C(saturation)
    elif mode == 'hsv':        
        ratio = [histT[4]/histC[4]-histT[2]/histC[2], histT[5]/histC[5]-histT[3]/histC[3]]
             
    # Gray模式：ratio = T_gray/C_gray
    elif mode == 'gray':        
        ratio = [histT[0]/histC[0], histT[1]/histC[1]]
               
    return (ratio)


def write_txt(dir_path,outlist,imgname,mode):
    
    classes = imgname.split('(')[0]
    filename = os.path.join(dir_path,mode+'_output.txt')
    with open(filename, 'a') as f:
        f.write(imgname+'\t')
        f.write(classes+'\t')
        for element in outlist:
            f.write(str(element)+'\t')
        f.write('\n')
        
        
def write_csv(imgname,filename,RGBlist,HSVlist,GRAYlist,IntensityR,IntensityG,IntensityB,IntensityS,IntensityV,IntensityGRAY):
    
    # classes = imgname.split('(')[0]  ############ hCG 0.1(1)(1)
    classes = imgname.split('-')[0]   ############ AFM/OTA 0.1-1
    # 列表中的rgb、hsv、灰度等值
    rgb = [str(element) for element in RGBlist[:-1]]
    hsv = [str(element) for element in HSVlist[:-1]]
    gray = [str(element) for element in GRAYlist[:-1]]

    with open(filename, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        # 构建一行数据
        row = [imgname, classes]+rgb+hsv+gray+IntensityR+IntensityG+IntensityB+IntensityS+IntensityV+IntensityGRAY
        csv_writer.writerow(row)



if __name__ == "__main__":
    
    # 图片储存路径   ######### hCG/AFM/OTA
    ROOT = 'D:\\MyProject\\hcg\\opencv'
    root_path1 = 'D:\\MyProject\\DatasetsOrig\\TestPaper-OTA'
    root_path2 = os.path.join(ROOT, 'data\\crop1')
    
    # 目标路径
    dir_path = os.path.join(ROOT, 'output\\exp')
    # 创建目标文件夹
    if os.path.exists(dir_path):
        print('文件夹 {}'.format(dir_path)+' 已存在，结果将保存至该文件夹中')
    else:
        os.mkdir(dir_path)
        print('创建文件夹 {}'.format(dir_path)+'，结果将保存至该文件夹中')
    # 输出文件表头初始化 (批量处理的时候可以屏蔽掉，防止文件覆盖)
    with open(os.path.join(dir_path,'rgb_output.txt'), 'w') as f:
        f.write('filename  class  T-red  C-red  T-green  C-green  T-blue  C-blue  ratio=T(red+blue)/C(red+blue)\n')
    with open(os.path.join(dir_path,'hsv_output.txt'), 'w') as f:
        f.write('filename  class  T-hue色相  C-hue  T-saturation饱和度  C-saturation  T-value明度  C-value  ratio=T(value)/C(value)-T(saturation)/C(saturation)\n')        
    with open(os.path.join(dir_path,'gray_output.txt'), 'w') as f:
        f.write('filename  class  T-gray  C-gray  ratio=T(gray)/C(gray)\n')

    ############### /AFM/OTA
    # filename = os.path.join(dir_path,'TestPaper_hCG.csv')
    # filename = os.path.join(dir_path,'TestPaper_AFM.csv')
    filename = os.path.join(dir_path,'TestPaper_OTA.csv')
    with open(filename, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['filename','class','Tr','Cr','Tg','Cg','Tb','Cb','Th','Ch','Ts','Cs','Tv','Cv','Tgray','Cgray',
                             'TintensR','CintensR','ratioR','TintensG','CintensG','ratioG','TintensB','CintensB','ratioB',
                             'TintensS','CintensS','ratioS','TintensV','CintensV','ratioV','TintensGRAY','CintensGRAY','ratioGRAY'])
       
    # 读取文件列表
    img_list = os.listdir(os.path.join(root_path1,'images'))  # 图像文件名列表
    lab_list = [] # 用于储存标签列表
    

    for img in img_list:
        name = os.path.splitext(img)[0]
        
        # 读取图像路径和标签路径 #################################
        img_path = os.path.join(root_path1,'images',img)  
        lab_path = os.path.join(root_path1,'labels',name+'.txt')
        image = cv2.imread(img_path, -1)  # 打开对应图片
        # plt.imshow(image)
        
        
        #######################################################################
        # 条带色度识别（裁剪）
        
        # 按照txt标签，裁剪出T线和C线
        img_T,contourT,img_C,contourC = img_crop(image,lab_path)
        # plt.imshow(img_T)
        # plt.imshow(img_C)
        """
        # 在原图绘制裁剪轮廓
        img0 = image.copy()
        img0[:,:,2], img0[:,:,0] = image[:,:,0], image[:,:,2]  # BGR2RGB
        contours = [contourT, contourC]
        contour_img = cv2.drawContours(img0, contours, -1, (0,255,0), 2)
        
        plt.figure() # 重新创建画布
        if image.shape[0] >= image.shape[1]:
            plt.subplot(221), plt.imshow(np.rot90(contour_img)), plt.title(name) #将试纸条横过来显示
        else:
            plt.subplot(221), plt.imshow(contour_img), plt.title(name)
        # plt.imshow(contour_img)
        # plt.savefig(os.path.join(dir_path,name+'.png'))
        """        
        ratios = {}  # (rgb_peak, rgb_area, hsv_peak, hsv_area, gray_peak, gray_area)
        
        # rgb颜色识别 ratio = T(red+blue)/C(red+blue)  越小越深
        path_rgb = os.path.join(dir_path,name+'_rgb')
        _,ratios['rgb'],rgb_output = extract_rgb(img_T,img_C,path_rgb)
        write_txt(dir_path,rgb_output,img,'rgb') # 将结果写入txt文件

        # hsv明度识别 ratio = T(value)/C(value)-T(saturation)/C(saturation)  越小越深
        path_hsv = os.path.join(dir_path,name+'_hsv')
        _,ratios['hsv'],hsv_output = extract_hsv(img_T,img_C,path_hsv)
        write_txt(dir_path,hsv_output,img,'hsv')
        
        # gray灰度识别 ratio = T_gray/C_gray  越小越深        
        path_gray = os.path.join(dir_path,name+'_gray')
        _,ratios['gray'],gray_output = extract_gray(img_T,img_C,path_gray)
        write_txt(dir_path,gray_output,img,'gray')

        
        #######################################################################
        # 整图灰度条带强度识别 （不裁剪）
        # 1. RGB识别
        path_intensity = os.path.join(dir_path,name+'_intensityR')
        red_intensity = extract_intensity(image,'Red',path_intensity)
        path_intensity = os.path.join(dir_path,name+'_intensityG')
        green_intensity = extract_intensity(image,'Green',path_intensity)
        path_intensity = os.path.join(dir_path,name+'_intensityB')
        blue_intensity = extract_intensity(image,'Blue',path_intensity)
        # 2. HSV识别
        path_intensity = os.path.join(dir_path,name+'_intensityS')
        saturation_intensity = extract_intensity(image,'Saturation',path_intensity)
        path_intensity = os.path.join(dir_path,name+'_intensityV')
        value_intensity = extract_intensity(image,'Value',path_intensity)
        # 3. 灰度识别
        path_intensity = os.path.join(dir_path,name+'_intensityGRAY')
        gray_intensity = extract_intensity(image,'Gray',path_intensity)
        
        #######################################################################

        
        # 写出总的输出结果(TestPaper.csv)，作为下一步回归用的数据
        write_csv(img,dir_path,rgb_output,hsv_output,gray_output,                  
                  red_intensity,green_intensity,blue_intensity,
                  saturation_intensity,value_intensity,gray_intensity)
        print (os.path.join(root_path1,'images',name)+' is done!')
        
        """
        # 绘图画布调整与图片储存
        plt.subplots_adjust(hspace=0.35)
        plt.savefig(os.path.join(dir_path,name+'.png'))
        plt.close()  # 关闭画布，释放缓存
        """