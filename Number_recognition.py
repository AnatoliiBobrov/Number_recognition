import time
import pickle
import torch
from torch.nn import (
    Module,
    Linear,
    MSELoss,
    Conv2d,
    ReLU,
    MaxPool2d,
    Dropout,
    Sequential
    )
from torch.nn import functional as F
from torch.optim import SGD

LEARNING_RATE = 0.05  # Learning rate
EPOCHES = 10  # Learning itterations
# 10 digit in tensor view
TARGETS = [torch.tensor([(0.01 if x != i else 0.99) for x in range (10)]) \
           for i in range(10)] 


class Net(Module):
    """
    Digit recognition network, MLP
    98,11% accuracy on test
    """
    def __init__(self):
        super(Net, self).__init__()
        self.L1 = Linear(28 * 28, 14 * 14)
        self.L2 = Linear(14 * 14, 7 * 7)
        self.L3 = Linear(7 * 7, 10)
        
    def forward(self, x):
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = self.L3(x)
        return x


class ConvNet(Module):
    """
    Digit recognition network, CNN + MLP 
    perceptron
    99,26% accuracy on test
    """
    def __init__(self, keep_prob):
        super(ConvNet, self).__init__()
        self.layer1 = Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=1 - keep_prob)
            )
        self.layer2 = Sequential(
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=1 - keep_prob)
            )
        self.layer3 = Sequential(
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2, padding=1),
            Dropout(p=1 - keep_prob)
            ) 
        self.fc1 = Linear(128 * 16, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) 
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape((128 * 16))
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def test_net(net, x_test, y_test):
    """
    Network survey
    """ 
    accuracy = 0
    begin = time.time()
    for i in range(len(x_test)):
        net_out = net(x_test[i])
        num = torch.argmax(net_out).item()
        target = y_test[i]
        if num == target:
            accuracy += 1 
    end = time.time()       
    t = end - begin
    accuracy = accuracy / len(x_test) * 100
    print (f"   Survey accuracy: {accuracy:.2f}%, time: {t:.2f}")  
    
def train_net(net, x_train, y_train, x_test, y_test):
    """
    Network training
    """
    optimizer = SGD(net.parameters(), lr=LEARNING_RATE)
    criterion = MSELoss() 
    b = time.time()
    for epoch in range (EPOCHES):
        accuracy = 0
        begin = time.time()
        for i in range(len(x_train)):
            net_out = net(x_train[i])
            num = torch.argmax(net_out).item()
            target = y_train[i]
            if num == target:
                accuracy += 1
            target = TARGETS[target]
            optimizer.zero_grad()
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
        end = time.time()       
        t = end - begin
        accuracy = accuracy / len(x_train) * 100
        print (f"   Epoch: {epoch}, accuracy: {accuracy:.2f}%, time: {t:.2f}")
        test_net(net, x_test, y_test)
    end = time.time()       
    t = end - b
    print (f"   Overall time: {t:.2f}")  

def get_data():
    """
    Return training and test data
    """
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    with open("data.bin", "rb") as file:
        (x_train, y_train), (x_test, y_test) = pickle.load(file)
    x_train_ = []
    for digit in x_train:
        x_train_.append(torch.tensor(digit.reshape((28*28))) / 255)
    x_test_ = []
    for digit in x_test:
        x_test_.append(torch.tensor(digit.reshape((28*28))) / 255)
    return x_train_, y_train, x_test_, y_test

def train_both():
    """
    Training of two types of NN
    """    
    # Training of fully connected perceptron
    net = Net()
    x_train, y_train, x_test, y_test = get_data()
    print ("Training of MLP...")
    train_net(net, x_train, y_train, x_test, y_test)

    # Training of convolutional NN + fully connected perceptron
    net = ConvNet(0.9)
    x_train_ = []
    for digit in x_train:
        x_train_.append(digit.reshape((1, 28, 28)))
    x_test_ = []
    for digit in x_test:
        x_test_.append(digit.reshape((1, 28, 28)))
    print ("Training of CNN + MLP...")
    train_net(net, x_train_, y_train, x_test_, y_test)
    
train_both()
