import matplotlib.pyplot as plt
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os
import pickle
opj = os.path.join
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import TrainAWD1
from AWD_model.DWT2d import DWT2d
choose_model_path = 'choose_result/wt.pth'
num_epochs = 3000
final_model = "final_model"
class SeparableConv2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: tuple, padding: tuple = 0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv2d(self.c_in, self.c_in, kernel_size=self.kernel_size,
                                        stride=1,padding=self.padding, groups=self.c_in, bias=False)
        self.conv2d_1x1 = nn.Conv2d(self.c_in, self.c_out, kernel_size=(22,1), stride=1,bias=False)
    def forward(self, x: torch.Tensor):
        y = self.depthwise_conv(x)
        y = self.conv2d_1x1(y)
        return y
# class Conv2dWithConstraint(nn.Conv2d):
#     def __init__(self, *args, max_norm=1, **kwargs):
#         self.max_norm = max_norm
#         super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
#
#     def forward(self, x):
#         self.weight.data = torch.renorm(
#             self.weight.data, p=2, dim=0, maxnorm=self.max_norm
#         )
#         return super(Conv2dWithConstraint, self).forward(x)
class DenseWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1.0,**kwargs):
        self.max_norm = max_norm
        super(DenseWithConstraint, self).__init__(bias=False,*args, **kwargs)
    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(DenseWithConstraint, self).forward(x)
class AWD_Model_Init(nn.Module):
    def __init__(self, wt):
        super(AWD_Model_Init, self).__init__()
        self.wt = wt.eval()
        self.wt.J = 3
        for param in wt.parameters():
            param.requires_grad = True
        self.Conv2_1 = SeparableConv2d(2,8,kernel_size=(1,5),padding=(0,1))
        self.Conv2_2 = SeparableConv2d(2,8,kernel_size=(1,3),padding=(0,0))
        self.bn3 = nn.BatchNorm2d(16,eps=1e-03,  affine=True,momentum=0.99)
        self.avg_pool = nn.AvgPool2d((1,4))
        self.fc1 = DenseWithConstraint(368, 5,max_norm=0.25)
        # self.fc2 = DenseWithConstraint(8, 5,max_norm=0.25)
    def forward(self, x):
        batch_size = x.shape[0]
        x_t = self.wt(x)
        x_t0 = x_t[0]
        x_t3 = x_t[3]

        x_1 = torch.cat([x_t0, x_t3],dim=1)

        x_1_1 = self.Conv2_1(x_1)
        x_1_2 = self.Conv2_2(x_1)
        # x_1_3 = self.Conv2_3(x_1)

        # x = torch.cat([x_1_1,x_1_2,x_1_3],dim=1)
        x = torch.cat([x_1_1,x_1_2],dim=1)


        x = F.elu(self.bn3(x))

        x = self.avg_pool(x)

        x = torch.flatten(x,1)

        x = self.fc1(x)
        # x = self.fc2(x)
        self.wt.J = self.wt.J
        return x
if __name__ == "__main__":
    with open(r'./data_path/train_sub01.pkl', 'rb') as f:
        train_loader = pickle.load(f)
    res = train_loader
    with open(r'./data_path/test_sub01.pkl', 'rb') as f:
        test_loader = pickle.load(f)
    wt = DWT2d(wave_row=TrainAWD1.wave_row, wave_column=TrainAWD1.wave_column, mode='periodization', J=TrainAWD1.J, init_factor=1, noise_factor=0.0).to(device)
    wt.load_state_dict(torch.load(choose_model_path))
    AWD_model = AWD_Model_Init(wt).to(device)
    optimizer = torch.optim.Adam(AWD_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    train_losses = []
    temp = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.
        for batch_idx, (data, y) in enumerate(train_loader):
            data = data.to(device)
            y = y.to(device).long()
            # zero grad
            optimizer.zero_grad()
            output = AWD_model(data)
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



    torch.save(AWD_model.state_dict(), opj(final_model, 'AWD_Model_单层FC.pth'))
    m = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    y_pred_AWD_model = np.zeros(m)
    y_true = np.zeros(m)
    with torch.no_grad():
        for batch_idx, (data, y) in tqdm(enumerate(test_loader, 0), total=int(np.ceil(m / batch_size))):
            data = data.to(device)
            # cnn prediction

            outputs_cnn = AWD_model(data)
            _, y_pred = torch.max(outputs_cnn.data, 1)
            y_pred_AWD_model[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y_pred.cpu().numpy()
            y_true[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y.numpy()
    print("AWD_model accuracy {:.2f}% ".format((y_true == y_pred_AWD_model).sum() / m * 100))
    plt.plot(train_losses)
    plt.show()
