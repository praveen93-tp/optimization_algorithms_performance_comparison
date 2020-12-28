import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import utils
from torch.nn import Parameter
import temp_config_files


class MLP(nn.Module):
    def __init__(self, num_features=784, num_hidden=[256,64,32], num_outputs=10):
        super(MLP, self).__init__()
        self.W_1 = Parameter(init.xavier_normal_(torch.Tensor(num_hidden[0], num_features)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_hidden[0]), 0))
        self.W_2 = Parameter(init.xavier_normal_(torch.Tensor(num_hidden[1], num_hidden[0])))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_hidden[1]), 0))
        self.W_3 = Parameter(init.xavier_normal_(torch.Tensor(num_hidden[2], num_hidden[1])))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_hidden[2]), 0))
        self.W_4 = Parameter(init.xavier_normal_(torch.Tensor(num_outputs, num_hidden[2])))
        self.b_4 = Parameter(init.constant_(torch.Tensor(num_outputs), 0))
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        x = x.view(x.size(0), -1)
        x = F.relu(F.linear(x, self.W_1, self.b_1))
        x = F.relu(F.linear(x, self.W_2, self.b_2))
        x = F.relu(F.linear(x, self.W_3, self.b_3))
        x = F.linear(x, self.W_4, self.b_4)
        return x



class CNN(nn.Module):
    def __init__(self, num_layers=3, num_filters=32, num_classes=10, input_size=(3, 32, 32)):
        super(CNN, self).__init__()
        self.channels = input_size[0]
        self.height = input_size[1]
        self.width = input_size[2]
        # print(self.width, self.height, self.channels, num_classes)
        self.num_filters = num_filters
        self.conv_in = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        # cnn = []
        # for _ in range(num_layers):
        #     cnn.append(nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, padding=1))
        #     cnn.append(nn.BatchNorm2d(self.num_filters))
        #     cnn.append(nn.ReLU(inplace = True))
        #     cnn.append(nn.MaxPool2d(kernel_size = 3, padding=1))
        # self.cnn = nn.Sequential(*cnn)
        # self.out_lin = nn.Linear(self.num_filters * self.width * self.height, num_classes)
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        x = F.relu(self.conv_in(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        # print(x.reshape(x.size(0),-1).shape)
        x = x.reshape(x.size(0), -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(x.shape)
        # x = self.cnn(x)
        # x = x.reshape(x.size(0), -1)
        # return self.out_lin(x)
        return x


def train(net, data, optimizer, batch_size=128, num_epochs=250, lr_schedule=False, hessian_free=False):
    train_generator = utils.data.DataLoader(data[0], batch_size=batch_size)
    val_generator = utils.data.DataLoader(data[1], batch_size=batch_size)
    losses = temp_config_files.AvgLoss()
    val_losses = temp_config_files.AvgLoss()
    plot_TrainLoss = []
    plot_ValidLoss = []
    Train_acc = []
    Valid_acc = []
    if lr_schedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    for epoch in range(num_epochs + 1):
        epoch_loss = temp_config_files.AvgLoss()
        epoch_val_loss = temp_config_files.AvgLoss()
        correct_val = 0
        correct_train = 0

        for x, y in val_generator:
            y = y.type(torch.LongTensor)
            if torch.cuda.is_available(): y = y.cuda()
            output = net(x)
            epoch_val_loss += F.cross_entropy(output, y).cpu()
            correct_val += (torch.max(output, 1)[1].view(y.size()).data == y.data).sum()
        accuracy_val = 100 * (correct_val.item() / len(val_generator.dataset))

        for x, y in train_generator:
            y = y.type(torch.LongTensor)
            if torch.cuda.is_available(): y = y.cuda()
            output = net(x)
            loss = F.cross_entropy(output, y).cpu()
            correct_train += (torch.max(output, 1)[1].view(y.size()).data == y.data).sum()
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy_train = 100 * (correct_train.item() / len(train_generator.dataset))

        if lr_schedule:
            scheduler.step(epoch_loss.avg)
        print(
            f'Epoch {epoch}/{num_epochs}, train loss: {epoch_loss}, val loss: {epoch_val_loss}, train acc: {accuracy_train}, val acc: {accuracy_val}')
        plot_TrainLoss.append(epoch_loss.avg)
        plot_ValidLoss.append(epoch_val_loss.avg)
        Train_acc.append(accuracy_train)
        Valid_acc.append(accuracy_val)

        losses += epoch_loss.losses
        val_losses += epoch_val_loss.losses
    return losses, val_losses, plot_TrainLoss, plot_ValidLoss, Train_acc, Valid_acc