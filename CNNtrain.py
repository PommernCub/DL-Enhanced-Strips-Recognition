import cv2
from matplotlib import pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import math
import warnings
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

import torch.nn as nn
import torchvision.models as models

# 忽略警告
warnings.filterwarnings('ignore')
import argparse
import datetime
import os.path
import random

np.random.seed(42)  # 设置numpy的随机种子
random.seed(42)  # 设置random模块的随机种子
torch.manual_seed(42)  # 为CPU设置随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)  #

def get_args():
    parser = argparse.ArgumentParser(description='浓度检测')
    parser.add_argument('--dataset', type=str, default="AFM",
                        help='填写需要进行训练的数据OTA, AFM')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0008, help='Learning rate')
    parser.add_argument('--model_name', type=str, default="efficientnet", help='选择的模型efficientnet, resnet50')
    parser.add_argument('--norm', type=str, default="contour_image", help='图片处理模式image_half, lower_half, contour_image')
    parser.add_argument('--augmentation', type=int, default=0, help='是否利用hcg图像进行增强0：不增强；1：增强')
    return parser.parse_args()

# 加载配置文件
args = get_args()


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

    # 图像增强：调整对比度和锐化
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image_enhanced = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    height = image_enhanced.shape[0]
    middle = height // 2
    lower_half = image_enhanced[middle:, :]
    image_half = image[middle:, :]
    # lower_half = image_enhanced[:, :]
    # image_half = image[:, :]
    gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = cv2.drawContours(lower_half.copy(), contours, -1, (0, 255, 0), 2)
    return image_half, lower_half, contour_image


def data_preprocessing_all(data_path):
    image_half_list, lower_half_list, contour_image_list, label_list = [], [], [], []
    for path in tqdm(data_path):
        image = cv2.imread(path)
        image_half, lower_half, contour_image = data_preprocessing_half(image)
        image_half = reshape_images(image_half, shape=(224, 224))
        lower_half = reshape_images(lower_half, shape=(224, 224))
        contour_image = reshape_images(contour_image, shape=(224, 224))

        image_half_list.append(image_half)
        lower_half_list.append(lower_half)
        contour_image_list.append(contour_image)
    return np.array(image_half_list), np.array(lower_half_list), np.array(contour_image_list)

# 数据加载和预处理
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return torch.tensor(image), torch.tensor(label)
path1 = "./data/AFM-TestPaper-crop/images"
path2 = "./data/OTA-TestPaper-crop/images"
path3 = "./data/hcg/images"

images_path1, label1 = patch_imagespath_label(path1)
images_path2, label2 = patch_imagespath_label(path2)
images_path3, label3 = patch_hcgimagespath_label(path3)

images_path1 = np.array(images_path1)
images_path2 = np.array(images_path2)
images_path3 = np.array(images_path3)

label1 = np.array(label1, dtype=float)
label2 = np.array(label2, dtype=float)
label3 = np.array(label3, dtype=int)
# 假设你有一个包含数据的数组 X 和对应标签的数组 y

# 使用hcg试纸图片进行数据增强
transformer_num_df = pd.read_excel("hCG.xlsx")
ordered_afm_concentrations_label3 = transformer_num_df.set_index('类别id').loc[label3]['对应AFM浓度'].tolist()
ordered_ota_concentrations_label3 = transformer_num_df.set_index('类别id').loc[label3]['对应OTA浓度'].tolist()

# X 和 y 应该是对应的，即 X 中的第 i 个样本对应 y 中的第 i 个标签

# 把数据集划分成训练集和剩余的部分
if args.dataset == "AFM":
    X_train, X_test, y_train, y_test = train_test_split(images_path1, label1, test_size=0.3, random_state=42)
elif args.dataset == "OTA":
    X_train, X_test, y_train, y_test = train_test_split(images_path2, label2, test_size=0.3, random_state=42)
else:
    X_train, X_test, y_train, y_test = None, None, None, None

if args.augmentation == 1:
    X_temp = np.concatenate([X_train, images_path3])
    y_temp = np.concatenate([y_train, ordered_afm_concentrations_label3])
else:
    X_temp, y_temp = X_train, y_train
# 再将剩余的部分划分成验证集和测试集
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42)
# X_train, y_train = X_temp, y_temp
# X_val, y_val = X_test, y_test
# 输出每个数据集的大小
print("训练集大小:", len(X_train))
print("验证集大小:", len(X_val))
print("测试集大小:", len(X_test))

X_train_image_half, X_train_lower_half, X_train_contour_image = data_preprocessing_all(X_train)
X_val_image_half, X_val_lower_half, X_val_contour_image = data_preprocessing_all(X_val)
X_test_image_half, X_test_lower_half, X_test_contour_image = data_preprocessing_all(X_test)

# 定义图像预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if args.norm == "image_half":
    # 假设 train_images 和 train_labels 分别是训练图像和标签的numpy数组
    train_dataset = CustomDataset(X_train_image_half, y_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # 假设 val_images 和 val_labels 分别是验证集图像和标签的numpy数组
    val_dataset = CustomDataset(X_val_image_half, y_val, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # 假设 val_images 和 val_labels 分别是验证集图像和标签的numpy数组
    test_dataset = CustomDataset(X_test_image_half, y_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
elif args.norm == "lower_half":
    # 假设 train_images 和 train_labels 分别是训练图像和标签的numpy数组
    train_dataset = CustomDataset(X_train_lower_half, y_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # 假设 val_images 和 val_labels 分别是验证集图像和标签的numpy数组
    val_dataset = CustomDataset(X_val_lower_half, y_val, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # 假设 val_images 和 val_labels 分别是验证集图像和标签的numpy数组
    test_dataset = CustomDataset(X_test_lower_half, y_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
elif args.norm == "contour_image":
    # 假设 train_images 和 train_labels 分别是训练图像和标签的numpy数组
    train_dataset = CustomDataset(X_train_contour_image, y_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # 假设 val_images 和 val_labels 分别是验证集图像和标签的numpy数组
    val_dataset = CustomDataset(X_val_contour_image, y_val, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # 假设 val_images 和 val_labels 分别是验证集图像和标签的numpy数组
    test_dataset = CustomDataset(X_test_contour_image, y_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
else:
    train_loader, val_loader, test_loader = None, None, None

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
            nn.Dropout(0.2),
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
            nn.Dropout(0.2),
            nn.Linear(self.num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)



if args.model_name == "efficientnet":
    model = RegressionEfficientNet()
elif args.model_name == "resnet50":
    model = RegressionResNet50()
else:
    model = None

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# 训练模型
num_epochs = args.epochs

# 检测CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型转移到设备
model.to(device)


# 更新训练和验证函数，确保数据也在正确的设备上
def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}  Train Loss: {running_loss / len(train_loader)}')


def validate(model, val_loader, criterion, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions.extend(outputs.squeeze().cpu().tolist())
            actuals.extend(labels.cpu().tolist())
    print()
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = math.sqrt(mse)
    try:
        mape = mean_absolute_percentage_error(actuals, predictions)
    except ValueError:
        mape = float('inf')
    # print(f'Validation MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}')
    return mape


def test_data(model, val_loader, criterion, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions.extend(outputs.squeeze().cpu().tolist())
            actuals.extend(labels.cpu().tolist())
    return actuals, predictions


# 初始化最佳MAPE为无限大
best_mape = float('inf')

model_save_path = os.path.dirname(os.path.abspath(__file__)) + "/output/model_path/" + args.dataset + "_" + args.model_name+ "_" + args.norm+ "_" + str(args.augmentation)+ "_" +'best_model.pth'
csv_save_path = os.path.dirname(os.path.abspath(__file__)) + "/output/results/" + args.dataset + "_" + args.model_name + "_" + args.norm + "_" + str(args.augmentation) + "_" +'result.csv'

print(csv_save_path)
for epoch in range(num_epochs):
    train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
    current_mape = validate(model, val_loader, criterion, device)
    # 检查是否是最佳MAPE
    if current_mape < best_mape:
        best_mape = current_mape
        # print(f'Saving best model with MAPE: {best_mape:.4f}')
        torch.save(model.state_dict(), model_save_path)

print(f'Saving best model with MAPE: {best_mape:.4f}')
model.load_state_dict(torch.load(model_save_path, map_location=device))
actuals, predictions = test_data(model, test_loader, criterion, device)
predictions = np.array(predictions)
predictions[predictions < 0] = 0
df = pd.DataFrame({"images_path": X_test, "actuals": actuals, "predictions": predictions})
df.to_csv(csv_save_path)
print(len(actuals))
