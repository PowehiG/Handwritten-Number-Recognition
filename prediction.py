import torch
import imgprocess as PRE
from train import MnistNet, test_loader
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 加载网络模型
m_state_dict = torch.load('./weight/mymodule.pt')
Net = MnistNet()
Net.load_state_dict(m_state_dict)


def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def plot_image1(img, label, name):
    fig = plt.figure()
    plt.tight_layout()
    plt.imshow(img[0]*0.3081+0.1307, cmap='gray', interpolation='none')
    plt.title("{}: {}".format(name, label.item()))
    plt.xticks([])
    plt.yticks([])
    plt.show()


def realpredict():
    # 对预测图片进行预处理
    img = PRE.image_preprocessing()
    # 将待预测图片转换形状
    inputs = img.reshape(-1, 784)
    inputs = torch.from_numpy(inputs)
    inputs = inputs.float()
    predict = Net(inputs)
    print("The number in this picture is {}".format(torch.argmax(predict).detach().numpy()))


def predict():
    x, y = next(iter(test_loader))
    out = Net(x.view(x.size(0), 28*28))
    pred = out.argmax(dim=1)
    plot_image(x, pred, 'test')


def confusion_matrix(preds, labels, conf_matrix):
    #preds = torch.argmax(preds, 1)

    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def testplot():
    conf_matrix = torch.zeros(10,10)   #创建矩阵
    for batch_images, batch_labels in test_loader:
        out = Net(batch_images.view(batch_images.size(0), 28*28))
        prediction = out.argmax(dim=1)
        conf_matrix = confusion_matrix(prediction, labels=batch_labels, conf_matrix=conf_matrix)

    conf = np.array(conf_matrix)
    print(conf)
    number = 10             #数字数量
    labels = ['0','1','2','3','4','5','6','7','8','9']  #数字对应标签

    plt.imshow(conf, cmap=plt.cm.Blues)

    thresh = conf.max() / 2 # 数值颜色阈值，如果数值超过这个，就颜色加深
    for x in range(number):
        for y in range(number):
            info = int(conf[y,x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color='white' if info > thresh else "black")
    plt.tight_layout()  # 保证图不重叠
    plt.yticks(range(number),labels)
    plt.xticks(range(number),labels)
    plt.show()
    plt.close()



if __name__ == '__main__':
    realpredict()