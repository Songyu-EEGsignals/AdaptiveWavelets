import matplotlib.pyplot as plt
import numpy as np
import torch,os,torchvision
import torch.nn.functional as F
import torch.nn.init as init

from torch import nn
from AWD_model.EEGNet import EEGNet

from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
opj = os.path.join
#----------------------------
data_path = "./data_path"
premodel_path = "./result_premodel"
premodel_name = "EEGNet.pth"
batch_size = 32
num_epochs = 80

#----------------------------
#Data_def:
#模仿tensorflow初始化权重方法
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                init.zeros_(m.bias.data)
        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                init.zeros_(m.bias.data)




if __name__ == "__main__":

    import pickle

    with open(r'./data_path/train_sub01.pkl', 'rb') as f:
        train_loader = pickle.load(f)
    with open(r'./data_path/test_sub01.pkl', 'rb') as f:
        test_loader = pickle.load(f)


    # import models
    EEGNet = EEGNet().to(device)
    initialize_weights(EEGNet)

    optimizer = torch.optim.Adam(EEGNet.parameters(), lr=0.001, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min=0, last_epoch=-1)

    criterion = torch.nn.CrossEntropyLoss()
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.
        for batch_idx, (data, y) in enumerate(train_loader):
            data = data.to(device)
            y = y.to(device)
            # zero grad
            optimizer.zero_grad()
            output = EEGNet(data)
            loss = criterion(output, y)
            # backward
            loss.backward()
            # update step
            optimizer.step()
            iter_loss = loss.item()
            epoch_loss += iter_loss
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), iter_loss), end='')
        mean_epoch_loss = epoch_loss / (batch_idx + 1)
        train_losses.append(mean_epoch_loss)
    # save model
    torch.save(EEGNet.state_dict(), opj(premodel_path, premodel_name))
    m = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    y_pred_cnn = np.zeros(m)
    y_true = np.zeros(m)
    with torch.no_grad():
        for batch_idx, (data, y) in tqdm(enumerate(test_loader, 0), total=int(np.ceil(m / batch_size))):
            data = data.to(device)
            # cnn prediction
            outputs_cnn = EEGNet(data)
            _, y_pred = torch.max(outputs_cnn.data, 1)
            y_pred_cnn[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y_pred.cpu().numpy()
           # labels
            y_true[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y.numpy()
    print("CNN accuracy {:.5f}%".format((y_true == y_pred_cnn).sum() / m * 100))
    plt.plot(train_losses)
