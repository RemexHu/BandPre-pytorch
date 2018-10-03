import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt
import pandas
import padasip as pa
import Train_fromRLS
import Train_pytorch

MODEL_CATEGORY = ['bus', 'car', 'ferry', 'metro', 'train', 'tram', 'general', '0.3general', '0.5general', '0.7general']
SET_CATEGOTY = ['bus', 'car', 'ferry', 'metro', 'train', 'tram']




DIR = '/home/runchen/Github/BandPre-pytorch/models/'
TEST_NUM = 3
TARGET_SIZE = 1
BATCH_SIZE = 96
TIME_STEP = 30
INPUT_SIZE = 1
HIDDEN_SIZE = 128



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=6,
            batch_first=True,
            dropout=0.15,
            #bidirectional=True
        )
        self.out = nn.Linear(HIDDEN_SIZE * TIME_STEP, TARGET_SIZE)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        # r_out shape (batch, time_step, hidden_size)
        # then do flattening in order to pass through the linear layer
        r_out = r_out.contiguous().view(BATCH_SIZE, 1, HIDDEN_SIZE * TIME_STEP)
        # print('1')
        # print(r_out.size())
        outs = self.out(r_out)
        # print(outs.size())
        return outs


def create_Dataset(dir):
    # Input: dir, path of log files
    # Output: dataset, flattened numpy array
    file = pandas.read_csv(dir, names=["band"], sep='\t')
    dataset = file["band"][:]
    dataset = np.asarray(dataset).astype(float)
    return dataset

def create_DataTensor(dataset, window):
    # Input: dataset, flattened numpy array
    # Output: Tensor_x [batch, time_step, features]
    #         Tensor_y [batch, target_size, features]
    # Each dataset has one output
    data_x, data_y = [], []
    for i in range(window, len(dataset)):
        batch_x = dataset[i - window: i]
        batch_y = dataset[i: i + TARGET_SIZE]
        data_x.append(batch_x[:, np.newaxis])
        data_y.append(np.array([batch_y]))

    data_x, data_y = np.asarray(data_x), np.asarray(data_y)

    Tensor_x = torch.from_numpy(data_x)
    Tensor_y = torch.from_numpy(data_y)

    return Tensor_x, Tensor_y

def create_DataLoader(Tensor_x, Tensor_y,):
    Tensor_batch = Data.TensorDataset(Tensor_x, Tensor_y)
    loader = Data.DataLoader(dataset=Tensor_batch, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)

    return loader

def test(model, DataLoader_test, loss_fn):
    loss_itr = []
    for loader in DataLoader_test:
        for (batch_x, batch_y) in loader:
            batch_x, batch_y = batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor)
            x, y = batch_x.cuda(), batch_y.cuda()
            # x, y = batch_x, batch_y
            output = model(x)
            loss_itr.append(loss_fn(output, y))

    return np.sum(sum(loss_itr).data.cpu().numpy())


def main():

    loss_table = [[0] * len(SET_CATEGOTY) for i in range(len(MODEL_CATEGORY))]
    loss_fn = nn.MSELoss()

    filedict_test = {'bus': 3, 'car': 3, 'ferry': 3, 'metro': 3, 'train': 3, 'tram': 3}
    testset = {}
    for category, num in filedict_test.items():
        for i in range(num):
            dir = 'test_sim_traces/norway_' + category + '_' + str(i + 1)
            testset[category] = testset.get(category, []) + [create_Dataset(dir)]

    for i, Mcategory in enumerate(MODEL_CATEGORY):
        model = torch.load(DIR + Mcategory + '.pkl')
        for j, Scategory in enumerate(SET_CATEGOTY):
            testing = testset[Scategory]
            DataLoader_test = []
            for dataset in testing:
                if dataset.shape[0] < TIME_STEP:
                    continue
                Tensor_x, Tensor_y = create_DataTensor(dataset, TIME_STEP)
                loader = create_DataLoader(Tensor_x, Tensor_y)
                DataLoader_test.append(loader)

            loss = test(model=model, DataLoader_test=DataLoader_test, loss_fn=loss_fn)
            loss_table[i][j] = loss
            print(Mcategory, Scategory)

    print(loss_table)

if __name__ == '__main__':
    main()
