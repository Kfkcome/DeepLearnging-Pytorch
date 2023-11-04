from dataset import MyDataset
from torch.autograd import Variable
import torch.nn as nn
import torch
import torchvision.models as models
from ResNet import resnet34
from torchvision import transforms
from VGG16 import VGG16
from Alexnet import AlexNet




def train_net(netname,net, device, data_path, validation_data_path, epochs=40, batch_size=8, lr=0.0001):
    # 加载训练集
    isbi_dataset = MyDataset(data_path, transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 加载验证集
    validation_dataset = MyDataset(validation_data_path, transform=data_transform["val"])
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)

    # 定义RMSprop算法
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # 定义Loss算法
    # 定义二元交叉熵函数
    # criterion = nn.BCELoss()
    # 定义损失函数，使用CrossEntropyLoss，即交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        print(netname+" ",epoch)
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        eval_loss = 0
        eval_acc = 0
        for image, label in train_loader:
            image, label = Variable(image), Variable(label)
            # 将数据拷贝到device中
            image = image.to(device=device)
            label = label.to(device=device)
            # 使用网络参数，输出预测结果
            pred = net(image)

            # 根据预测概率推得结果
            predicted = torch.max(pred, 1)[1]

            # 计算loss
            loss = criterion(pred, label)
            # 为后边计算总的loss和accu添加数据
            eval_loss += loss.item()
            test_correct = (predicted == label).sum()
            eval_acc += test_correct.item()

            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')

            # 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 计算训练集的准确率
        print('Train_loss: ', eval_loss / len(isbi_dataset), 'Train acc: ', eval_acc / len(isbi_dataset))
        # 验证模式
        net.eval()
        best_validation_loss = float('inf')
        validation_loss = 0.0
        test_correct = 0
        with torch.no_grad():
            for val_image, val_label in validation_loader:
                val_image = val_image.to(device=device)
                val_label = val_label.to(device=device)
                val_pred = net(val_image)
                predicted = torch.max(val_pred, 1)[1]
                val_loss = criterion(val_pred, val_label)
                test_correct += (predicted == val_label).sum().item()
                validation_loss += val_loss.item()

        average_validation_loss = validation_loss / len(validation_dataset)
        average_validation_acc = test_correct / len(validation_dataset)
        print('Loss/validation', average_validation_loss, 'Aucc/validation', average_validation_acc)

        # 如果当前验证损失更低，则保存模型
        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            torch.save(net.state_dict(), 'best_model.pth')
    net.eval()
    eval_loss = 0
    eval_acc = 0
    for step, data in enumerate(validation_loader):
        batch_x, batch_y = data
        batch_x, batch_y = Variable(batch_x).to(device), Variable(batch_y).to(device)
        out = net(batch_x)
        predicted = torch.max(out, 1)[1]
        loss = criterion(out, batch_y)
        eval_loss += loss.item()
        test_correct = (predicted == batch_y).sum()
        eval_acc += test_correct.item()
    print(netname+" "+'Test_loss: ', eval_loss / len(validation_dataset), 'Test acc: ', eval_acc / len(validation_dataset))
    print()

if __name__ == "__main__":
    ## 选择显卡
    torch.cuda.set_device(0)
    ## 数据预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((512, 512)),
            # # transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Resize(512),
            # transforms.CenterCrop(224),
            # transforms.ToTensor()
        ]),  # 来自官网参数
        "val": transforms.Compose([
            transforms.Resize((512,512)),
            # transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # 有cuda or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_model=[]

    # 定义自定义的ResNet模型
    # net = resnet34()
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 2)
    # net.to(device)  # 将网络迁移到GPU进行计算
    # all_model.append(["自定义的ResNet模型",net])

    # 使用自定义的VGG16模型
    # net=VGG16()
    # net.to(device)
    # all_model.append(["自定义的VGG16模型",net])

    # 使用自定义的AlexNet
    # net=AlexNet(num_classes=2).to(device)
    # all_model.append(["自定义的AlexNet",net])

    # # 使用包自带的inception_v3
    # model = models.inception_v3(pretrained=True)
    # num_cls = 2  # Number of classes
    # num_aux_fc = model.AuxLogits.fc.in_features
    # model.AuxLogits.fc = torch.nn.Linear(num_aux_fc, num_cls)
    # num_fc = model.fc.in_features
    # model.fc = torch.nn.Linear(num_fc, num_cls)
    # model.to(device)

    # 使用包自带ResNet网络
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # PyTorch的nn.Linear（）是用于设置网络中的全连接层的
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft.to(device=device)
    all_model.append(["包自带ResNet网络",model_ft])

    # # 使用VIT
    #
    # models = ViT(dim=128, image_size=512, patch_size=64, num_classes=2,
    #             transformer=data_transform["train"], channels=3).to(device)

    # 是使用包自带的预训练的vgg16
    # model = models.vgg16(pretrained=False)
    # num_cls = 2  # 分类数
    # num_fc = model.classifier[6].in_features
    # model.classifier[6] = torch.nn.Linear(num_fc, num_cls)
    # model.to(device=device)
    # all_model.append(["包自带的预训练的vgg16",model])

    #使用自带的预训练的alexnet
    # model_ft = models.alexnet(pretrained=True)
    # # num_ftrs = model_ft.fc.in_features
    # # PyTorch的nn.Linear（）是用于设置网络中的全连接层的
    # # model_ft.fc = nn.Linear(num_ftrs, 2)
    # model_ft.to(device=device)
    # all_model.append(["自带的预训练的alexnet",model_ft])

    # 指定训练集地址，开始训练
    data_path = "../data/train/"
    test_path = "../data/test/"
    # train_net(net, device, data_path, test_path, 30, 128)

    train_net("resnet",model_ft, device, data_path, test_path, 30, 128)
