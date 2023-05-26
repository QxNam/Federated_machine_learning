import numpy as np
import pandas as pd
import cv2
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import flwr as fl
from collections import OrderedDict
import sys

import color
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cores = mp.cpu_count()
c = color.clr()
print(c.TEXT_BOLD('='*80))
print(c.SUCCESS('Day:'), datetime.datetime.now())
print(c.SUCCESS('Device:'), device)
print(c.SUCCESS('Core:'), cores)
print(c.TEXT_BOLD('='*80))

BATCH_SIZE = 16
SIZE_IMAGE = (227, 227)

# --- Data ---
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.Resize(SIZE_IMAGE)])

def pre_process(img):
    '''
    pre-process image before training
    algorithm: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    '''
    img = img.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    return img_clahe

class Dataset():
    def __init__(self, path_data, transform=None):
        self.path_data = path_data.upper()
        self.df = pd.read_csv(f'dataset/{self.path_data}-ROI-Mammography/description.csv')
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image_path = f'dataset/{self.path_data}-ROI-Mammography/{self.df.iloc[index].Path_save}'
        image = cv2.imread(image_path, 0)
        if self.transform:
            image = pre_process(image)
            image = self.transform(image)
        label = self.df.iloc[index]['Cancer']
        return image, label
    
def train_test_split(dataset, test_size=0.2):
    s_test = int(test_size * dataset.__len__())
    s_train = dataset.__len__() - s_test
    train_dataset, test_dataset = random_split(dataset, [s_train, s_test])
    return train_dataset, test_dataset
    
def dataloader(train_dataset, test_dataset, batch_size=16):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

dataset = Dataset('MIAS', transform=transform)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
train_dataloader, test_dataloader = dataloader(train_dataset, test_dataset, batch_size=BATCH_SIZE)
# --- Model ---
# google net
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out1x1, reduce3x3, out3x3, reduce5x5, out5x5, out1x1pool):
        super(InceptionModule, self).__init__()

        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out1x1, kernel_size=1)

        # 1x1 convolution followed by 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce3x3, out3x3, kernel_size=3, padding=1)
        )

        # 1x1 convolution followed by 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce5x5, out5x5, kernel_size=5, padding=2)
        )

        # 3x3 max pooling followed by 1x1 convolution branch
        self.branch1x1pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out1x1pool, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch1x1pool = self.branch1x1pool(x)
        outputs = [branch1x1, branch3x3, branch5x5, branch1x1pool]
        return torch.cat(outputs, 1)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(GoogLeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

model = GoogLeNet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# --- Define Flower client ---
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [param.detach().numpy() for param in model.parameters()]

    def set_parameters(self, parameters):
        params_dict = zip(model.parameters(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.train()
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1).float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        return self.get_parameters(), len(train_dataloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()
        total_loss = 0.0
        total_correct = 0
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                outputs = model(inputs.unsqueeze(1).float())
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
        accuracy = total_correct / len(test_dataset)
        return total_loss, len(test_dataset), {"accuracy": accuracy}

    def set_parameters(self, parameters):
        idx = 0
        for param in model.parameters():
            param.data = torch.from_numpy(parameters[idx])
            idx += 1
# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)