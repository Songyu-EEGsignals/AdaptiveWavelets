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
from AWD_model.DWT1d import DWT1d
from AWD_model.DWT2d import DWT2d

choose_model_path = 'choose_result/wt.pth'
num_epochs = 20
final_model = "final_model"





class AWD_Model_Init(nn.Module):
    def __init__(self, wt):
        super(AWD_Model_Init, self).__init__()
        self.fc1 = nn.Linear(18048, 1024)
        self.fc2 = nn.Linear(1024, 5)

        self.wt = wt.eval()
        self.wt.J = 3
        # freeze layers
        for param in wt.parameters():
            param.requires_grad = False

    def forward(self, x):
        batch_size = x.shape[0]
        x_t = self.wt(x)




        x_t0 = x_t[0]
        x_t1 = x_t[1]
        x_t2 = x_t[2]
        x_t3 = x_t[3]

        self.wt.J = 2
        x_t1 = self.wt(F.relu(x_t1))
        self.wt.J = 1
        x_t2 = self.wt(F.relu(x_t2))

        x = []
        x.append(x_t0.reshape(batch_size, -1))
        for j in range(len(x_t1)):
            x.append(x_t1[j].reshape(batch_size, -1))
        for j in range(len(x_t2)):
            x.append(x_t2[j].reshape(batch_size, -1))
        x.append(x_t3.reshape(batch_size, -1))
        x = torch.cat(x, 1)


        x = F.elu(self.fc1(x))
        x = self.fc2(x)


        self.wt.J = 3
        return x

if __name__ == "__main__":

    with open(r'./data_path/train_sub01.pkl', 'rb') as f:
        train_loader = pickle.load(f)
    with open(r'./data_path/test_sub01.pkl', 'rb') as f:
        test_loader = pickle.load(f)
    wt = DWT2d(wave_row=TrainAWD1.wave_row, mode='periodization', J=TrainAWD1.J, init_factor=1, noise_factor=0.0).to(device)
    wt.load_state_dict(torch.load(choose_model_path))
    AWD_model = AWD_Model_Init(wt).to(device)
    optimizer = torch.optim.Adam(AWD_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []

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
    # torch.save(AWD_model.state_dict(), opj(final_model, 'AWD_Model.pth'))
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
    print("AWD_model accuracy {:.5f}% ".format((y_true == y_pred_AWD_model).sum() / m * 100))
    plt.plot(train_losses)
    plt.show()
