# # python imports
# import os
# #from tqdm import tqdm

# # torch imports
# import torch
# import torch.nn as nn
# import torch.optim as optim

# # helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(255, 255), num_classes=1081):
        super(LeNet, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input.)
        self.conv1 = nn.Conv2d( in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1 )
        # Max-pooling
        self.max_pool_1 = nn.MaxPool2d(kernel_size = 2, stride = 2 )
        # Convolution
        self.conv2 = nn.Conv2d( in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1 )
        # Max-pooling
        self.max_pool_2 = nn.MaxPool2d( kernel_size = 2, stride = 2 )
        # Fully connected layer
        self.fc1 = nn.Linear( in_features = 6 * 1028, out_features = 4 * 1028 )
        # Fully connected layer
        self.fc2 = nn.Linear( in_features = 4 * 1028, out_features = 2 * 1028 )
        # Fully connected layer
        self.fc3 = nn.Linear( in_features = 2 * 1028, out_features = num_classes )

    def forward(self, x):
        shape_dict = {}
        # convlve, then perform ReLU non-linearity
        x = nn.functional.relu(self.conv1(x))
        # max pooling with 2x2 grid
        x = self.max_pool_1(x)
        shape_dict[1] = list(x.size());
        # convlve, then perform ReLU non-linearity
        x = nn.functional.relu(self.conv2(x))
        # max pooling with 2x2 grid
        x = self.max_pool_2(x)
        shape_dict[2] = list(x.size());
        # flatten
        x = x.view(-1, 400) 
        shape_dict[3] = list(x.size());
        # fully connected layer, then perform ReLU non-linearity
        x = nn.functional.relu(self.fc1(x))
        shape_dict[4] = list(x.size());
        # fully connected layer, then perform ReLU non-linearity
        x = nn.functional.relu(self.fc2(x))
        shape_dict[5] = list(x.size());
        # fully connected layer
        x = self.fc3(x)
        shape_dict[6] = list(x.size());
        return x, shape_dict

def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input, target = input.to(device), target.to(device)
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################
        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            input, target = input.to(device), target.to(device)
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc


if __name__ == "__main__":
    print(torchvision.list_models())