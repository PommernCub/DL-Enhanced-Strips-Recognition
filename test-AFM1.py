import warnings

import torch
import numpy as np
import cv2
from torchvision import transforms, models
import torch.nn as nn
import argparse
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
warnings.filterwarnings('ignore')


class RegressionEfficientNet(nn.Module):
    def __init__(self, num_classes=1):  # num_classes=1 for regression
        super(RegressionEfficientNet, self).__init__()
        # 加载预训练的EfficientNet，不包括最后的全连接层
        self.base_model = models.efficientnet_b0(pretrained=True)
        # 获取特征提取层的输出特征数
        self.num_ftrs = self.base_model.classifier[
            1].in_features  # Access the in_features of the Linear layer inside the classifier
        # 替换分类层以适用于回归任务
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.01),
            nn.Linear(self.num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

class RegressionResNet50(nn.Module):
    def __init__(self, num_classes=1):  # num_classes=1 for regression
        super(RegressionResNet50, self).__init__()
        # 加载预训练的ResNet50，不包括最后的全连接层
        self.base_model = models.resnet50(pretrained=True)
        # 获取特征提取层的输出特征数
        self.num_ftrs = self.base_model.fc.in_features  # Access the in_features of the Linear layer inside the fc (fully connected) layer
        # 替换分类层以适用于回归任务
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)



def load_model(model_path, model_name):
    if model_name == "efficientnet":
        model = RegressionEfficientNet()
    elif model_name == "resnet50":
        model = RegressionResNet50()
    else:
        raise ValueError("Unsupported model name")

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def reshape_images(original_image, shape=(224, 224)):
    # 读取图像
    # 调整裁剪后图像的尺寸到224x224
    # cv2.INTER_LINEAR是插值方法，适用于放大图像
    resized_image = cv2.resize(original_image, shape, interpolation=cv2.INTER_LINEAR)
    return resized_image


def patch_imagespath_label(path):
    images_list = os.listdir(path)
    images_list = [file for file in images_list if file.endswith('.png') or file.endswith('.jpg')]

    label = []
    images_path = []
    for image in images_list:
        label.append(image.split("-")[0])
        images_path.append(os.path.join(path, image))
    return images_path, label


def patch_hcgimagespath_label(path):
    images_list = os.listdir(path)
    images_list = [file for file in images_list if file.endswith('.png') or file.endswith('.jpg')]

    label = []
    images_path = []
    for image in images_list:
        label.append(image.split("(")[0])
        images_path.append(os.path.join(path, image))
    return images_path, label


def data_preprocessing_half(image):

    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image_enhanced = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    height = image_enhanced.shape[0]
    middle = height // 2
    lower_half = image_enhanced[middle:, :]
    image_half = image[middle:, :]
    gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = cv2.drawContours(lower_half.copy(), contours, -1, (0, 255, 0), 2)
    return image_half, lower_half, contour_image

# 定义图像预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def preprocess_image(data_path, args):
    image_half_list, lower_half_list, contour_image_list, label_list = [], [], [], []
    image = cv2.imread(data_path)
    image_half, lower_half, contour_image = data_preprocessing_half(image)
    image_half = reshape_images(image_half, shape=(224, 224))
    lower_half = reshape_images(lower_half, shape=(224, 224))
    contour_image = reshape_images(contour_image, shape=(224, 224))

    image_half_list.append(image_half)
    lower_half_list.append(lower_half)
    contour_image_list.append(contour_image)
    if args.norm == "contour_image":
        return np.array(contour_image_list)
    elif args.norm == "image_half":
        return np.array(image_half_list)
    elif args.norm == "lower_half":
        return np.array(lower_half_list)

def predict(model, image):
    model.eval()

    with torch.no_grad():
        output = model(image)

        return output.squeeze().cpu().tolist()

def calculate_precision(data):
    
    mse_grouped = mean_squared_error(data['actuals'], data['predictions'])
    rmse_grouped = mean_squared_error(data['actuals'], data['predictions'], squared=False)
    mae_grouped = mean_absolute_error(data['actuals'], data['predictions'])
    r2_grouped = r2_score(data['actuals'], data['predictions'])
    return pd.Series({'MSE': mse_grouped, 'RMSE': rmse_grouped, 'MAE': mae_grouped, 'R-square': r2_grouped})

def main():
    parser = argparse.ArgumentParser(description='Predict concentration from an image')
    parser.add_argument('--images_dir', type=str, default="data/AFM1-test"
                        , help='Path to the image file')
    parser.add_argument('--model_path', type=str, default="./output-best/model_path/AFM_efficientnet_lower_half_0_best_model.pth", help='Path to the trained model file')
    parser.add_argument('--model_name', type=str, default="efficientnet", help='Model name (efficientnet or resnet50)')
    parser.add_argument('--norm', type=str, default="lower_half", help='图片处理模式image_half, lower_half, contour_image')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    print(args.images_dir)
    model = load_model(args.model_path, args.model_name)
    model.to(device)
    csv_save_path = 'result-test-AFM.csv'
    df = pd.DataFrame(columns=["images_path", "actuals", "predictions"])
 
    # Iterate over all images in the directory
    for filename in os.listdir(args.images_dir):
        image_path = os.path.join(args.images_dir, filename)
        if os.path.isfile(image_path):
            print(f"Processing {image_path}")
            
            image = preprocess_image(image_path, args)
            image = image[0]
            image = transform(image)
            image = torch.tensor(image).to(device)
            image = image.unsqueeze(0)
            prediction = predict(model, image)
            if prediction < 0:
                prediction = 0
        
            print(f"Predicted concentration: {prediction}")
            df = df.append({"images_path": filename, "actuals": filename.split('-')[0], "predictions": prediction}, ignore_index=True)
            df.to_csv(csv_save_path)
            
    # 评估预测值的准确度
    results = pd.read_csv(csv_save_path)
    
    grouped_precision = results.groupby('actuals').apply(calculate_precision).reset_index()
    mse = mean_squared_error(results['actuals'], results['predictions'])
    rmse = mean_squared_error(results['actuals'], results['predictions'], squared=False)
    mae = mean_absolute_error(results['actuals'], results['predictions'])
    r2 = r2_score(results['actuals'], results['predictions'])
    
    print(grouped_precision)
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R-squared: {r2}")

if __name__ == "__main__":
    main()
