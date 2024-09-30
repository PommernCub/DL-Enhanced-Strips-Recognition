
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
1. 灰度处理
2. 进行灰度/RGB/HSV色度识别
3. 输出 (图片分析和色度数据) 结果文件

"""

def extract_intensity(image,mode,save_path):

    if mode == 'Gray':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)            

    # 对图片进行裁剪
    height, width, _ = image.shape
    left = width * 0.1
    right = width - left
    img = img[:, int(left):int(right)]
    if (image.shape[0] >= image.shape[1]):
        y1,y2,x1,x2 = int(0.02*img.shape[0]),int(0.98*img.shape[0]),int(0.02*img.shape[1]),int(0.98*img.shape[1])
        img = img[y1:y2, x1:x2]
        imgL,imgW = img.shape[0],img.shape[1]
        intensity_profile = 255-np.mean(img, axis=1) # 纵向摆放的试纸条            
    else:
        y1,y2,x1,x2 = int(0.02*img.shape[0]),int(0.98*img.shape[0]),int(0.02*img.shape[1]),int(0.98*img.shape[1])
        img = img[y1:y2, x1:x2]
        imgL,imgW = img.shape[1],img.shape[0]
        intensity_profile = 255-np.mean(img, axis=0) # 横向摆放的试纸条


    # 使用高斯滤波器平滑数据        
    intensity_profile = gaussian_filter(intensity_profile, sigma=2) # sigma越大曲线越平滑，不然取不到准确峰信息
    with open(f'Gray_{name}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f'{name}'])  # 写入标题行
        for intensity in intensity_profile:
            writer.writerow([intensity])
    # 找出峰的位置
    peaks, heights = find_peaks(intensity_profile, prominence=0.8)
    # 绘图-RGB模式
    plt.close('all')
    print (name+':\t'+str(peaks))
    img_rgb = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    plt.imshow(cv2.transpose(img_rgb), extent=[0,imgL,60,imgW+60])
    plt.plot(intensity_profile, label=name, color='#1772b4')
    plt.plot(peaks, intensity_profile[peaks], "o", color='#f0a1a8', label='Peaks')
    plt.title(name+' Intensity Profile - '+mode)
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    plt.show()
    plt.close('all')
    # peak不为两个时额外处理
    if (len(peaks) != 2):

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
                
                # 去掉C线后面多余的峰，手动决定去掉哪个峰                
                selected_peaks = [peaks[0]]  # 创建一个列表以保存用户选择的元素（里面有第一个峰的索引0）
                # 输入想要保留的C峰的索引
                for i in range(1):
                    while True:
                        try:
                            index = int(input(f"图中从左到右，哪个峰是T线的峰（1 到 {len(peaks)}）: "))-1
                            if 0 <= index < len(peaks):
                                selected_peaks.append(peaks[index])
                                break  # 有效索引，跳出循环
                            else:
                                print("输入的索引超出范围，请重新输入。")
                        except ValueError:
                            print("输入无效，请输入一个整数索引。")
                peaks = selected_peaks
                
                # # 不想手动，自动选取最高的两个峰
                # peak_index = heights['prominences'].argsort()[-2:][::-1]  # 按峰高降序排列
                # peaks = peaks[peak_index]
                # heights['prominences'] = heights['prominences'][:2]
        """          
        # 重新作图
        plt.imshow(img.T, extent=[0,imgL,70,imgW+70])
        plt.plot(intensity_profile, label=name, color='#ed5a65')
        plt.plot(peaks, intensity_profile[peaks], "x", label='Peaks')
        plt.title(name+' Intensity Profile - '+mode)            
        plt.show()
        """
        
    # 由峰的位置来计算宽度和边界
    rel_h = [0.97, 0.97]  # 两个峰取不同的高度
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
        width_props = peak_widths(intensity_profile, [peaks[0]], rel_height=0.95)
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
        # plt.plot(peaks, intensity_profile[peaks], "x", label='Peaks')
        plt.plot(peaks, intensity_profile[peaks], "o", color='#f0a1a8', label='Peaks')
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

        
def write_csv(imgname,dir_path,IntensityGRAY):
    
    # classes = imgname.split('(')[0]  ############ hCG 0.1(1)(1)
    classes = imgname.split('-')[0]   ############ AFM/OTA 0.1-1
        
    ############### /AFM/OTA
    # filename = os.path.join(dir_path,'TestPaper_hCG.csv')
    # filename = os.path.join(dir_path,'TestPaper_AFM1.csv')
    filename = os.path.join(dir_path,'TestPaper_OTA.csv')
    with open(filename, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        # 构建一行数据
        row = [imgname, classes]+IntensityGRAY
        csv_writer.writerow(row)



if __name__ == "__main__":
    
    # 图片储存路径   ######### hCG/AFM/OTA
    ROOT = 'D:\\MyProject\\hCG\\opencv'
    # root_path1 = 'D:\\MyProject\\DatasetsOrig\\AFM-Test'
    root_path1 = 'D:\\MyProject\\DatasetsOrig\\OTA-Test' ################ 图片存放地址
    root_path2 = os.path.join(ROOT, 'data\\TestImg') ################ 图片存放地址
    
    # 目标路径
    dir_path = os.path.join(ROOT, 'output\\exp')
    # 创建目标文件夹
    if os.path.exists(dir_path):
        print('文件夹 {}'.format(dir_path)+' 已存在，结果将保存至该文件夹中')
    else:
        os.mkdir(dir_path)
        print('创建文件夹 {}'.format(dir_path)+'，结果将保存至该文件夹中')

    ############# hCG/AFM/OTA
    # with open(os.path.join(dir_path,'TestPaper_hCG.csv'), 'w', newline='') as f:
    # with open(os.path.join(dir_path,'TestPaper_AFM.csv'), 'w', newline='') as f:
    with open(os.path.join(dir_path,'TestPaper_OTA.csv'), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['filename','class','T-GRAY','C-GRAY','ratioGRAY'])
       
    # 读取文件列表
    img_list = os.listdir(root_path2)  # 图像文件名列表
    lab_list = []  # 用于储存标签列表
    

    for img in img_list:
        name = os.path.splitext(img)[0]
        
        # 读取图像路径和标签路径
        # img_path = os.path.join(root_path1,'images1',img)
        img_path = os.path.join(root_path2,img)  ################ 图片存放地址
        # lab_path = os.path.join(root_path1,'labels',name+'.txt')
        image = cv2.imread(img_path, -1)  # 打开对应图片
        plt.close('all')
        plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        
        
        #######################################################################
        # 整图灰度条带强度识别 （不裁剪）
        path_intensity = os.path.join(dir_path,name+'_intensityGRAY')
        gray_intensity = extract_intensity(image,'Gray',path_intensity)
        
        #######################################################################
        
        # 写输出结果(TestPaper.csv)
        write_csv(img,dir_path,gray_intensity)
        print (os.path.join(root_path2,name)+' is done!') ################ 图片存放地址！
        """
        # 绘图画布调整与图片储存
        plt.subplots_adjust(hspace=0.35)
        plt.savefig(os.path.join(dir_path,name+'.png'))
        plt.close()  # 关闭画布，释放缓存
        """