import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

import matplotlib.pyplot as plt
import pandas
import padasip as pa

cuda = torch.device('cuda')

TIME_STEP = 30
TARGET_SIZE = 1
INPUT_SIZE = 1
HIDDEN_SIZE = 128

BATCH_SIZE = 96
LR = 0.0002
EPOCH = 20
RLS_MU = 0.6
IS_RLS = False


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


def create_RLSTensor(RLS_y):
    data_y = []
    for i in range(len(RLS_y)):
        batch_y = RLS_y[i: i + TARGET_SIZE]
        data_y.append(np.array([batch_y]))

    data_y = np.asarray(data_y)
    Tensor_y = torch.from_numpy(data_y)

    return Tensor_y


def create_DataLoader_RLS(Tensor_x, Tensor_y, Tensor_RLS):
    Tensor_batch = Data.TensorDataset(Tensor_x, Tensor_y, Tensor_RLS)
    loader = Data.DataLoader(dataset=Tensor_batch, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)

    return loader


def create_DataLoader(Tensor_x, Tensor_y,):
    Tensor_batch = Data.TensorDataset(Tensor_x, Tensor_y)
    loader = Data.DataLoader(dataset=Tensor_batch, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)

    return loader


def create_RLSdataset(dataset):
    # Input: dataset, flattened numpy array
    # Output: (RLS_x, RLS_y), list of RLS tensor
    RLS_x, RLS_y = [], []
    for i in range(TIME_STEP, len(dataset)):
        batch_x = dataset[i - TIME_STEP: i]
        batch_y = dataset[i: i + TARGET_SIZE]
        RLS_x.append(batch_x)
        RLS_y.append(np.array(batch_y))

    return RLS_x, RLS_y


def RLS_prediction(RLS_x, RLS_y):
    RLS = pa.filters.FilterRLS(TIME_STEP, RLS_MU)
    pred = np.zeros(len(RLS_x))

    for cur in range(len(RLS_x)):
        prediction = RLS.predict(RLS_x[cur])
        RLS.adapt(RLS_y[cur], RLS_x[cur])
        pred[cur] = prediction

    return pred


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


class Net_RLS(nn.Module):
    def __init__(self):
        super(Net_RLS, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=6,
            batch_first=True,
            dropout=0.15,
            # bidirectional=True
        )
        self.out = nn.Linear(64, TARGET_SIZE)
        self.hidden = nn.Linear(2, 64)
        self.out_lstm = nn.Linear(HIDDEN_SIZE * TIME_STEP, TARGET_SIZE)

    def forward(self, x, output_RLS):
        r_out, (h_n, h_c) = self.lstm(x, None)
        # r_out shape (batch, time_step, hidden_size)
        # then do flattening in order to pass through the linear layer
        r_out = r_out.contiguous().view(BATCH_SIZE, 1, HIDDEN_SIZE * TIME_STEP)
        # print('1')
        # print(r_out.size())
        output_lstm = self.out_lstm(r_out)
        # print(outs.size())

        output = torch.cat((output_lstm, output_RLS), dim=2)
        hidden = self.hidden(output)
        outs = self.out(hidden)


        return outs


def train(model, model_RLS, DataLoader_train, DataLoader_test, epochs, optimizer, loss_fn):
    loss_curve = []

    for epoch in range(epochs):
        itr = 0

        if IS_RLS:
            for loader in DataLoader_train:
                for (batch_x, batch_y, batch_RLS) in loader:
                    batch_x, batch_y, batch_RLS = batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor), batch_RLS.type(torch.FloatTensor)
                    batch_x, batch_y, batch_RLS = batch_x.cuda(), batch_y.cuda(), batch_RLS.cuda()
                    # x, y = Variable(batch_x), Variable(batch_y)
                    # output_RLS = Variable(batch_RLS)

                    output = model_RLS(batch_x, batch_RLS)

                    loss = loss_fn(output, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    itr += 1

                    if itr % 50 == 0:
                        loss_val = val_RLS(model_RLS=model_RLS, DataLoader_test=DataLoader_test, loss_fn=loss_fn)
                        loss_curve.append(loss_val)
                        print('Epoch = {} Iter = {} val_loss = {}'.format(epoch, itr, loss_val))



        else:
            for loader in DataLoader_train:
                for (batch_x, batch_y) in loader:
                    batch_x, batch_y = batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor)
                    x, y = batch_x.cuda(), batch_y.cuda()
                    # x, y = Variable(batch_x), Variable(batch_y)
                    output = model(x)

                    loss = loss_fn(output, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    itr += 1

                    if itr % 50 == 0:
                        loss_val = val(model=model, DataLoader_test=DataLoader_test, loss_fn=loss_fn)
                        loss_curve.append(loss_val)
                        print('Epoch = {} Iter = {} val_loss = {}'.format(epoch, itr, loss_val))


    loss_array = np.asarray(loss_curve)
    loss_array = np.hstack(loss_array)

    return model, loss_array


def val(model, DataLoader_test, loss_fn):
    loss_itr = []
    for loader in DataLoader_test:
        for (batch_x, batch_y) in loader:
            batch_x, batch_y = batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor)
            x, y = batch_x.cuda(), batch_y.cuda()
            # x, y = batch_x, batch_y
            output = model(x)
            loss_itr.append(loss_fn(output, y))

    return np.sum(sum(loss_itr).data.cpu().numpy())


def val_RLS(model_RLS, DataLoader_test, loss_fn):
    loss_itr = []
    for loader in DataLoader_test:
        for (batch_x, batch_y, batch_RLS) in loader:
            batch_x, batch_y, batch_RLS = batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor), batch_RLS.type(torch.FloatTensor)
            batch_x, batch_y, batch_RLS = batch_x.cuda(), batch_y.cuda(), batch_RLS.cuda()

            # x, y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
            # x, y = Variable(batch_x), Variable(batch_y)
            # output_RLS = Variable(batch_RLS)

            output = model_RLS(batch_x, batch_RLS)
            loss_itr.append(loss_fn(output, batch_y))

    return np.sum(sum(loss_itr).data.cpu().numpy())



def main():

    filedict_train_1 = {'bus': 11, 'car': 5, 'ferry': 15, 'metro': 16, 'train': 4, 'tram': 17}
    trainset = {}
    for category, num in filedict_train_1.items():
        for i in range(num):
            dir = 'train_sim_traces/' + category + str(i) +'.log'
            trainset[category] = trainset.get(category, []) + [create_Dataset(dir)]

    filedict_train_2 = {'bus': 24, 'car': 13, 'ferry': 21, 'metro': 10, 'train': 22, 'tram': 57}
    for category, num in filedict_train_2.items():
        for i in range(3, num):
            dir = 'test_sim_traces/norway_' + category + '_' + str(i)
            trainset[category] = trainset.get(category, []) + [create_Dataset(dir)]


    filedict_test = {'bus': 3, 'car': 3, 'ferry': 3, 'metro': 3, 'train': 3, 'tram': 3}
    testset = {}
    for category, num in filedict_test.items():
        for i in range(num):
            dir = 'test_sim_traces/norway_' + category + '_' + str(i + 1)
            testset[category] = testset.get(category, []) + [create_Dataset(dir)]


    for category in trainset:
        training = trainset[category]
        testing = testset[category]


        if IS_RLS:
            DataLoader_train, DataLoader_test = [], []
            for dataset in training:
                if dataset.shape[0] < TIME_STEP:
                    continue
                RLS_x, RLS_y = create_RLSdataset(dataset)
                pred = RLS_prediction(RLS_x, RLS_y)

                Tensor_x, Tensor_y = create_DataTensor(dataset, TIME_STEP)
                Tensor_RLS = create_RLSTensor(pred)

                loader = create_DataLoader_RLS(Tensor_x, Tensor_y, Tensor_RLS)
                DataLoader_train.append(loader)
            for dataset in testing:
                if dataset.shape[0] < TIME_STEP:
                    continue
                RLS_x, RLS_y = create_RLSdataset(dataset)
                pred = RLS_prediction(RLS_x, RLS_y)

                Tensor_x, Tensor_y = create_DataTensor(dataset, TIME_STEP)
                Tensor_RLS = create_RLSTensor(pred)

                loader = create_DataLoader_RLS(Tensor_x, Tensor_y, Tensor_RLS)
                DataLoader_test.append(loader)


        else:
            DataLoader_train, DataLoader_test = [], []
            for dataset in training:
                if dataset.shape[0] < TIME_STEP:
                    continue
                Tensor_x, Tensor_y = create_DataTensor(dataset, TIME_STEP)
                loader = create_DataLoader(Tensor_x, Tensor_y)
                DataLoader_train.append(loader)
            for dataset in testing:
                if dataset.shape[0] < TIME_STEP:
                    continue
                Tensor_x, Tensor_y = create_DataTensor(dataset, TIME_STEP)
                loader = create_DataLoader(Tensor_x, Tensor_y)
                DataLoader_test.append(loader)


        BandwidthLSTM = Net()
        BandwidthLSTM_RLS = Net_RLS()
        BandwidthLSTM.cuda()
        BandwidthLSTM_RLS.cuda()

        if IS_RLS:
            optimizer = optim.Adam(BandwidthLSTM_RLS.parameters(), lr=LR)
        else:
            optimizer = optim.Adam(BandwidthLSTM.parameters(), lr=LR)

        loss_fn = nn.MSELoss()

        model, loss_curve = train(model=BandwidthLSTM,
                                  model_RLS=BandwidthLSTM_RLS,
                                  DataLoader_train=DataLoader_train,
                                  DataLoader_test=DataLoader_test,
                                  epochs=EPOCH,
                                  optimizer=optimizer,
                                  loss_fn=loss_fn)

        torch.save(model, '/home/runchen/Github/BandPre-pytorch/models/' + category + '.pkl')

        plt.figure(figsize=(25, 9))
        plt.plot(loss_curve)
        plt.savefig('/home/runchen/Github/BandPre-pytorch/Curves/Loss_curve_' + category + '.png')
        # plt.show()


if __name__ == '__main__':
    main()