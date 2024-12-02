import torch
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *


Learning_Rate = 1e-2
Epochs = 50
total_train_step = 0
total_test_step = 0




dataset_Train = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
dataset_Val= torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False,
                                            transform=torchvision.transforms.ToTensor())

train_data_size = len(dataset_Train)
val_data_size = len(dataset_Val)


train_dataloader = DataLoader(dataset_Train, batch_size=64, shuffle=True)
val_dataloader = DataLoader(dataset_Val, batch_size=64 , shuffle=True)

model = Model()
model.cuda()

loss_fn = nn.CrossEntropyLoss()
loss_fn.cuda()


optimizer = torch.optim.SGD(model.parameters(), lr=Learning_Rate, )

writer = SummaryWriter(log_dir='./logs')

for i in range(Epochs):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)


    # 验证开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in val_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / val_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / val_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(model, "model_{}.pth".format(i))
    print("模型已保存")



writer.close()