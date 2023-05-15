import torch
import numpy as np
import argparse
from torch import nn
import  torch.nn.functional as F
def inference(model_path, path):
    images1, labels1 = convert_dataset(path)
    arr = []
    batch_size = 200
    net = Net()
    net.load_state_dict(torch.load(model_path))
    norm_images = normalize(images1)
    with torch.no_grad():
        for i in range(len(norm_images)//batch_size):
            batch = i * batch
            data = torch.from_numpy(norm_images[batch:batch + batch_size]).float()
            data = data.view(-1,28*28)
            output = net(data)
            _, predicted = torch.max(output.data, 1)
            arr.append(predicted)
    return arr


def normalize(im):
    imin = float(im.min())
    imax = float(im.max())
    return (im - imin)/(imax - imin)


def convert_dataset(train_path):
    dataset = np.genfromtxt(train_path, delimiter=',', skip_header=1)
    labels = dataset[:, 0].astype(np.uint8)
    values = dataset[:, 1:].astype(np.uint8)
    images = np.reshape(values, (-1, 28, 28))
    return images, labels

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=5,stride=1, padding=2)
        self.maxpool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(6*6*64,1000)
        self.fc2 = nn.Linear(1000, 10)
        # self.fc1 = nn.Linear(28*28, 200)
        # self.fc2 = nn.Linear(200, 200)
        # self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.maxpool(out)
        out = out.reshape(x.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return out

def main():
    parser = argparse.ArgumentParser(description="Inference CLI")
    parser.add_argument("model_path", type=str,  help = "model.pth")
    parser.add_argument("dataset_path", type=str, help = "Path to the input dataset")
    args = parser.parse_args()
    inference(args.model_path, args.image_path)
if __name__ == '__main__':
    main()
#%%
