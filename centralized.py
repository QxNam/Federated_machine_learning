import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split

import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
import color
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cores = mp.cpu_count()
c = color.clr()


def pre_process(img):
    img = img.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    return img_clahe
# --- Data ---
class Dataset():
    def __init__(self, path_data, transform=None):
        self.path_data = path_data
        self.df = pd.read_csv(f'{self.path_data}/description.csv')
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image_path = f'{self.path_data}/{self.df.iloc[index].Path_save}'
        image = cv2.imread(image_path, 0)
        if self.transform:
            image = pre_process(image)
            image_trans = self.transform(image)#['image']
        else:
            image_trans = image
        label = self.df.iloc[index]['Cancer']
        return image_trans, label
    
def Dataloader(path_data, transform=None, batch_size=16, size_train=0.8):
    dataset = Dataset(path_data=path_data, transform=transform)
    train_size = int(size_train * dataset.__len__())
    test_size = dataset.__len__() - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

def show_image(dataset):
    img = dataset[0][0].squeeze()
    label = dataset[0][1]
    cv2.imshow(str(label.item()), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Model ---
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train(net, trainloader, testloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    # net.train()
    lst_loss = {'train':[], 'val':[]}
    lst_acc = {'train':[], 'val':[]}
    correct, total, epoch_loss = 0, 0, 0.0
    print('-'*50)
    for epoch in range(epochs):
        print(c.SUCCESS('Epoch:'), epoch+1)
        loop = tqdm(trainloader, desc='Training')
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            acc_cur = (torch.max(outputs.data, 1)[1] == labels).sum().item()/labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=acc_cur)
        epoch_loss /= len(trainloader.dataset)
        train_acc = correct / total
        print(c.TEXT_BOLD('Train:'), f"loss {epoch_loss:.4f}, accuracy {train_acc:.4f}")
        lst_loss['train'].append(epoch_loss.item())
        lst_acc['train'].append(train_acc)
        # Validation
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        net.eval()
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            loss /= len(testloader.dataset)
            val_acc = correct / total
            print(c.TEXT_BOLD('Validation:'),f"loss {loss:.4f}, accuracy {val_acc:.4f}")
            lst_acc['val'].append(val_acc)
            lst_loss['val'].append(loss)
    return lst_loss, lst_acc

def transform_image():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomRotation(30),
                                    transforms.Resize((227,227))])
    return transform
def client(name_client):
    
    train_loader, test_loader = Dataloader(path_data=f'dataset/{name_client.upper()}-ROI-Mammography', \
        transform=transform_image(), batch_size=32)
    # Training
    net = Net().to(device)
    # show
    lst_loss, lst_acc = train(net, train_loader, test_loader, 5)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(lst_loss['train'], label='train')
    plt.plot(lst_loss['val'], label='val')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(lst_acc['train'], label='train')
    plt.plot(lst_acc['val'], label='val')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(f'result/{name_client}.png')

if __name__ == '__main__':
    print(c.TEXT_BOLD('MACHINE LEARNING ON CENTRALIZED DATA'))
    print(c.SUCCESS('Day:'), datetime.datetime.now())
    print(c.SUCCESS('Device:'), device)
    print(c.SUCCESS('Core:'), cores)

    # client('mias')
    net = Net().to(device)
    print(net)


