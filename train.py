import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

# 下载训练集
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),  # 图片转换函数
                               download=True)
# 下载测试集
test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)
# 设置批次
batch_size = 100
# 装载训练集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)  # 打乱数据集
# 装载测试集
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


# 定义网络
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()  # 继承父类
        self.Conn_layers = nn.Sequential(
            nn.Linear(784, 100),  # 网络输入层
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),  # 网络输出层
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        output = self.Conn_layers(input)
        return output


# 学习率
LR = 0.1
# 网络对象
net = MnistNet()
# 损失函数,交叉熵
loss_function = nn.CrossEntropyLoss()
# 优化函数使用SGD
optimizer = optim.SGD(
    net.parameters(),
    lr=LR,
    momentum=0.9,
    weight_decay=0.0005
)


def train(epoch):
    for epoch in range(epoch):
        for i, data in enumerate(train_loader):  # 枚举类
            inputs, labels = data
            inputs = inputs.reshape(batch_size, 784)  # 以100*784的结构输入
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()  # 梯度置0
            loss.backward()  # 进行反向传播，计算题都
            optimizer.step()  # 参数更新


def test(epoch):
    # 用测试数据进行测试
    test_result = 0
    for data_test in test_loader:
        images, labels = data_test

        # 转换输入形状
        images = images.reshape(batch_size, 784)
        images, labels = Variable(images), Variable(labels)
        output_test = net(images)

        # 对一个批次的数据的准确性进行判断
        for i in range(len(labels)):
            # 如果输出结果的最大值的缩影与标签内正确数据相等，准确个数累加
            if torch.argmax(output_test[i]) == labels[i]:
                test_result += 1
    print("Epoch {} : {} / {}".format(epoch, test_result, len(test_dataset)))


if __name__ == '__main__':
    for i in range(10):
        train(i)
        test(i)
    torch.save(net.state_dict(), './weight/mymodule.pt')
