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

LEARNING_RATE = 0.05 # Learning rate
EPOCHES = 1 # Learning itterations
TARGETS = [] # 10 digit in tensor view

for i in range (10):
    l_target = [(0.01 if x != i else 0.99) for x in range (10)]
    TARGETS.append(torch.tensor(l_target))


class ConvNet(Module):
    """
    Digit recognition network, convolutional NN + fully connected 
    perceptron
    99,20% accuracy on test
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
  




# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# Data_2 - there are data in tensor (28*28) from 0 to 1

"""
x_train_ = []
for i in range(len(x_train)):
    x_train_.append(torch.tensor(x_train[i]) / 255)
x_test_ = []
for i in range(len(x_test)):
    x_test_.append(torch.tensor(x_test[i]) / 255)
pickle.dump(((x_train_, y_train), (x_test_, y_test)), open("data_2.bin", "wb"))
raise Exception(3466)

"""


class Net(Module):
    """
    Digit recognition network, fully connected perceptron
    97,95% accuracy on test
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
    print (f"   Survey accuracy: {accuracy:.2f}, time: {t:.2f}")
   
    
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


def train_both():
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    with open("data_2.bin", "rb") as file:
        (x_train, y_train), (x_test, y_test) = pickle.load(file)
        
    # Training of fully connected perceptron
    net = Net()
    print ("Training of fully connected perceptron...")
    train_net(net, x_train, y_train, x_test, y_test)

    # Training of convolutional NN + fully connected perceptron
    net = ConvNet(0.9)
    x_train_ = []
    for digit in x_train:
        x_train_.append(digit.reshape((1, 28, 28)))
    x_test_ = []
    for digit in x_test:
        x_test_.append(digit.reshape((1, 28, 28)))
    print ("Training of convolutional NN + fully connected perceptron...")
    train_net(net, x_train_, y_train, x_test_, y_test)
    
train_both()
"""
train 2 variants?
add first launch function 

x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_test = x_test.reshape(x_test.shape[0], 28 * 28)

pickle.dump (((x_train, y_train), (x_test, y_test)), open("data_1.bin", "wb"))
Brute force
Epoch: 0, accuracy: 56259, time: 119.67
Epoch: 1, accuracy: 58463, time: 117.88
Epoch: 2, accuracy: 58928, time: 116.80
Epoch: 3, accuracy: 59166, time: 114.92
Epoch: 4, accuracy: 59347, time: 114.36
Epoch: 5, accuracy: 59453, time: 114.72
Epoch: 6, accuracy: 59530, time: 114.19
Epoch: 7, accuracy: 59598, time: 114.06
Epoch: 8, accuracy: 59660, time: 114.75
Epoch: 9, accuracy: 59723, time: 114.99

Accuracy: 9795, time: 3.31 - recognition accuracy of network

Epoch: 0, accuracy: 56371, time: 119.97
Survey accuracy: 9667, time: 2.77
Epoch: 1, accuracy: 58514, time: 120.49
Survey accuracy: 9748, time: 3.00
Epoch: 2, accuracy: 58918, time: 128.11
Survey accuracy: 9782, time: 2.80
Epoch: 3, accuracy: 59187, time: 126.71
Survey accuracy: 9793, time: 2.66
Epoch: 4, accuracy: 59377, time: 126.59
Survey accuracy: 9802, time: 4.08
Epoch: 5, accuracy: 59486, time: 139.50
Survey accuracy: 9811, time: 3.03
Epoch: 6, accuracy: 59576, time: 172.83
Survey accuracy: 9818, time: 4.63

Epoch: 0, accuracy: 56230, time: 112.60
Survey accuracy: 9643, time: 2.17
Epoch: 1, accuracy: 58472, time: 113.35
Survey accuracy: 9735, time: 2.17
Epoch: 2, accuracy: 58915, time: 116.97
Survey accuracy: 9776, time: 2.56
Epoch: 3, accuracy: 59195, time: 130.80
Survey accuracy: 9786, time: 2.36
Epoch: 4, accuracy: 59367, time: 141.48
Survey accuracy: 9803, time: 3.04
Epoch: 5, accuracy: 59485, time: 124.92
Survey accuracy: 9804, time: 2.31
Epoch: 6, accuracy: 59567, time: 134.01
Survey accuracy: 9809, time: 2.44
Epoch: 7, accuracy: 59636, time: 135.54
Survey accuracy: 9812, time: 2.35
Epoch: 8, accuracy: 59694, time: 108.31
Survey accuracy: 9806, time: 2.13
Epoch: 9, accuracy: 59741, time: 129.94
Survey accuracy: 9815, time: 2.74
Epoch: 10, accuracy: 59775, time: 168.75
Survey accuracy: 9818, time: 2.53
Epoch: 11, accuracy: 59802, time: 144.97
Survey accuracy: 9816, time: 2.94


Epoch: 0, accuracy: 57519, time: 513.73
Survey accuracy: 9829, time: 30.65
Epoch: 1, accuracy: 59117, time: 551.12
Survey accuracy: 9880, time: 32.52
Epoch: 2, accuracy: 59264, time: 543.13
Survey accuracy: 9884, time: 41.19
Epoch: 3, accuracy: 59361, time: 517.59
Survey accuracy: 9905, time: 27.94
Epoch: 4, accuracy: 59436, time: 482.33
Survey accuracy: 9910, time: 27.33
Epoch: 5, accuracy: 59471, time: 477.14
Survey accuracy: 9915, time: 26.76
Epoch: 6, accuracy: 59537, time: 474.90
Survey accuracy: 9910, time: 26.97
Epoch: 7, accuracy: 59551, time: 475.28
Survey accuracy: 9910, time: 26.93
Epoch: 8, accuracy: 59568, time: 474.89
Survey accuracy: 9907, time: 26.90
Epoch: 9, accuracy: 59620, time: 477.36
Survey accuracy: 9917, time: 28.79
Epoch: 10, accuracy: 59607, time: 474.58
Survey accuracy: 9929, time: 26.60
Epoch: 11, accuracy: 59639, time: 470.87
Survey accuracy: 9926, time: 26.60
Epoch: 12, accuracy: 59659, time: 472.05
Survey accuracy: 9925, time: 26.76
Epoch: 13, accuracy: 59682, time: 470.78
Survey accuracy: 9925, time: 27.13
Epoch: 14, accuracy: 59691, time: 471.07
Survey accuracy: 9926, time: 26.62
Epoch: 15, accuracy: 59705, time: 471.42
Survey accuracy: 9929, time: 26.76
Epoch: 16, accuracy: 59712, time: 471.37
Survey accuracy: 9937, time: 26.70
Epoch: 17, accuracy: 59741, time: 472.01
Survey accuracy: 9931, time: 26.69
Epoch: 18, accuracy: 59733, time: 471.20
Survey accuracy: 9923, time: 26.61
Epoch: 19, accuracy: 59731, time: 471.70
Survey accuracy: 9920, time: 26.66
Epoch: 20, accuracy: 59743, time: 472.41
Survey accuracy: 9931, time: 26.78
Epoch: 21, accuracy: 59762, time: 470.98
Survey accuracy: 9924, time: 26.62
Epoch: 22, accuracy: 59756, time: 470.85
Survey accuracy: 9931, time: 26.69
Epoch: 23, accuracy: 59772, time: 471.00
Survey accuracy: 9927, time: 26.68
Epoch: 24, accuracy: 59760, time: 471.12
Survey accuracy: 9927, time: 26.59
Epoch: 25, accuracy: 59773, time: 470.33
Survey accuracy: 9932, time: 26.67
Epoch: 26, accuracy: 59782, time: 470.58
Survey accuracy: 9927, time: 26.72
Epoch: 27, accuracy: 59787, time: 472.24
Survey accuracy: 9931, time: 26.57
Epoch: 28, accuracy: 59806, time: 470.46
Survey accuracy: 9923, time: 26.64
Epoch: 29, accuracy: 59815, time: 470.40
Survey accuracy: 9927, time: 26.81
Epoch: 30, accuracy: 59790, time: 471.11
Survey accuracy: 9938, time: 26.57
Epoch: 31, accuracy: 59818, time: 472.92
Survey accuracy: 9922, time: 26.62
Epoch: 32, accuracy: 59803, time: 470.80
Survey accuracy: 9924, time: 26.61
Epoch: 33, accuracy: 59809, time: 470.12
Survey accuracy: 9924, time: 26.65
Epoch: 34, accuracy: 59816, time: 477.58
Survey accuracy: 9935, time: 26.61
Epoch: 35, accuracy: 59838, time: 470.58
Survey accuracy: 9929, time: 26.63
Epoch: 36, accuracy: 59832, time: 470.36
Survey accuracy: 9929, time: 26.63
Epoch: 37, accuracy: 59823, time: 470.12
Survey accuracy: 9938, time: 26.55
Epoch: 38, accuracy: 59833, time: 470.48
Survey accuracy: 9938, time: 26.61
Epoch: 39, accuracy: 59826, time: 470.27
Survey accuracy: 9929, time: 26.72
Epoch: 40, accuracy: 59846, time: 469.52
Survey accuracy: 9929, time: 26.72
Epoch: 41, accuracy: 59850, time: 470.95
Survey accuracy: 9932, time: 26.62
Epoch: 42, accuracy: 59839, time: 469.50
Survey accuracy: 9939, time: 27.15
Epoch: 43, accuracy: 59847, time: 470.27
Survey accuracy: 9927, time: 26.66

new lr 0.01
Epoch: 0, accuracy: 57748, time: 380.05
Survey accuracy: 9845, time: 21.22
Epoch: 1, accuracy: 59132, time: 377.54
Survey accuracy: 9876, time: 21.80
Epoch: 2, accuracy: 59310, time: 385.55
Survey accuracy: 9888, time: 22.55
Epoch: 3, accuracy: 59388, time: 397.28
Survey accuracy: 9877, time: 21.19
Epoch: 4, accuracy: 59464, time: 387.23
Survey accuracy: 9905, time: 23.86
Epoch: 5, accuracy: 59520, time: 417.01
Survey accuracy: 9910, time: 22.08


Epoch: 0, accuracy: 55269, time: 595.38
Survey accuracy: 9673, time: 24.94
Epoch: 1, accuracy: 58426, time: 490.12
Survey accuracy: 9769, time: 23.27
Epoch: 2, accuracy: 58786, time: 456.40
Survey accuracy: 9842, time: 32.25

Epoch: 0, accuracy: 57339, time: 486.17
Survey accuracy: 9816, time: 27.36
Epoch: 1, accuracy: 59050, time: 477.83
Survey accuracy: 9870, time: 26.81
Epoch: 2, accuracy: 59282, time: 476.24
Survey accuracy: 9894, time: 27.04
Epoch: 3, accuracy: 59353, time: 474.79
Survey accuracy: 9885, time: 26.77
Epoch: 4, accuracy: 59426, time: 474.67
Survey accuracy: 9890, time: 26.82
Epoch: 5, accuracy: 59461, time: 478.70
Survey accuracy: 9907, time: 26.81
Epoch: 6, accuracy: 59512, time: 474.79
Survey accuracy: 9909, time: 26.80
Epoch: 7, accuracy: 59532, time: 483.23
Survey accuracy: 9910, time: 26.83
Epoch: 8, accuracy: 59565, time: 475.98
Survey accuracy: 9911, time: 26.83
Epoch: 9, accuracy: 59593, time: 535.83
Survey accuracy: 9920, time: 41.42
Epoch: 10, accuracy: 59615, time: 503.29
Survey accuracy: 9922, time: 28.10
Epoch: 11, accuracy: 59632, time: 524.24
Survey accuracy: 9926, time: 29.96
Epoch: 12, accuracy: 59672, time: 494.22
Survey accuracy: 9921, time: 27.68
Epoch: 13, accuracy: 59675, time: 529.18
Survey accuracy: 9927, time: 29.97
Epoch: 14, accuracy: 59673, time: 477.19
Survey accuracy: 9914, time: 26.61
"""