from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_train_val_index(input_all, nsample, network=[]):
    s = random.randint(0, 100000)
    if network:
        if nsample > 0:
            train_index, val_index, selected_station = select_station_samples(input_all, network, nstation=nsample,
                                                                              seed=s)
            out = [train_index, val_index, selected_station]
        else:
            train_index, val_index, selected_station = select_one(input_all, network, seed=s)
            out = [train_index, val_index, selected_station]
    else:
        if nsample > 1:
            train_index, val_index = evenly_selected_samples(input_all, nsample=nsample, seed=s, y=2021)
        else:
            all_index = np.asarray(input_all.index)
            (train_index, val_index, _, _) = train_test_split(all_index, all_index, train_size=nsample, random_state=s)
        out = [train_index, val_index]

    return out


def select_station_samples(input_all, network_name, nstation=1, seed=1):
    input_train = input_all[input_all['network'].isin(network_name)]
    val_index = input_train.index
    # input_train = select_stations_4_allyear(input_train)
    yy = input_train.groupby('station').count()['lc'].sample(n=nstation, random_state=seed)

    selected_station = yy.index.tolist()
    train_index = input_train[input_train['station'].isin(selected_station)].index
    # val_index = list(np.setdiff1d(np.asarray(input_all.index), train_index))
    return train_index, val_index, selected_station


def select_one(input_all, network_name, seed=1):
    input_train = input_all[input_all['network'].isin(network_name)]
    val_index = input_train.index
    input_train = select_stations_4_allyear(input_train)
    yy = input_train.groupby('station').count()['lc']

    selected_station = [yy.index.tolist()[seed]]
    train_index = input_train[input_train['station'].isin(selected_station)].index
    # val_index = list(np.setdiff1d(np.asarray(input_all.index), train_index))
    return train_index, val_index, selected_station


def select_stations_4_allyear(input_all):
    count_all = []
    edate = [0, 42735, 43100, 43465, 43830, 44196, 44561]
    for i in range(6):
        temp_index = np.asarray(input_all['edate'] > edate[i]) & np.asarray(input_all['edate'] <= edate[i + 1])
        temp_input = input_all.loc[temp_index, :]
        count_all.append(temp_input.groupby('station').count()['lc'])
    annual_sta = pd.concat(count_all, axis=1)
    annual_flag = np.sum(annual_sta.fillna(0) > 0, axis=1)
    yy = annual_flag[annual_flag > 4]  # confirm the year required
    train_station = yy.index.tolist()
    return input_all[input_all['station'].isin(train_station)]


def evenly_selected_samples(input_all, nsample=1, seed=1, y=2021):
    def subset_data_year(input_all, y=2016):
        edate = [0, 42735, 43100, 43465, 43830, 44196, 44561]
        edate_y = list(range(2016, 2022))
        idx = edate_y.index(y)
        selected_index = np.asarray(input_all['edate'] > edate[idx]) & np.asarray(input_all['edate'] <= edate[idx + 1])
        input_train = input_all.loc[selected_index]
        input_val = input_all.loc[~selected_index]
        return input_train, input_val, selected_index

    input_train, _, _ = subset_data_year(input_all, y=y)
    train_station = input_train.groupby('station').count()['lc']
    yy = train_station[train_station > 10]
    train_station = [yy.index[x] for x in range(len(yy.index))]
    selected_index = []
    for i in range(len(train_station)):
        station_df = input_train[input_train['station'] == train_station[i]]
        selected_index.append(station_df.sample(n=nsample, random_state=seed).index)
    train_index = np.asarray(selected_index).flatten()
    val_index = list(np.setdiff1d(np.asarray(input_all.index), train_index))
    return train_index, val_index


def prepare_train_val_data_multiscale(samples_part, train_val_index, samples_9km,batch_size=128,br=1):
    xtrain, ytrain = split_train_val(samples_part, train_val_index[0])
    xtest, ytest = split_train_val(samples_part, train_val_index[1])

    x9km, y9km = split_train_val(samples_9km)

    # train loader
    train_ds = CustomDataset(xtrain, ytrain)
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # target loader
    target_ds = CustomDataset(xtest, ytest)
    targetloader = DataLoader(target_ds, batch_size=batch_size*br, shuffle=True, num_workers=0)

    # target loader
    ds_9km = CustomDataset(x9km, y9km)
    loader_9km = DataLoader(ds_9km, batch_size=batch_size * br, shuffle=True, num_workers=0)

    # prepare testing set
    val_data = test_set_GPU(xtest, ytest)
    train_data = test_set_GPU(xtrain, ytrain)
    return trainloader, val_data, targetloader, train_data, loader_9km


def split_train_val(samples_part, train_index=[]):
    if len(train_index)==0:
        xtrain = np.asarray(samples_part.loc[:, '0':'152']).astype(np.float32)
        xtrain = input_reorganize(xtrain)
        ytrain = np.asarray(samples_part.loc[:, 'gt']).astype(np.float32)
    else:
        xtrain = np.asarray(samples_part.loc[train_index, '0':'152']).astype(np.float32)
        xtrain = input_reorganize(xtrain)
        ytrain = np.asarray(samples_part.loc[train_index, 'gt']).astype(np.float32)
    return xtrain, ytrain


def input_reorganize(x):
    d=x.shape[1]
    x1 = x[:,0:15]
    x2 = x[:,15:d]
    x2_mean=np.reshape(x2,[x2.shape[0],int(d/46),46])
    x1=np.concatenate([x1,np.nanmean(x2_mean,axis=2),x2],axis=1)
    #x1=np.concatenate([x1,x2_mean[:,:,45],x2],axis=1)
    #x2= np.transpose(x2,(0,2,1))
    return x1


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, scale_data=False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        #self.input_reorganize()
        if scale_data:
            self.X = StandardScaler().fit_transform(X)
        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y).reshape((self.y.shape[0], 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]  # X[i,:] is also OK

    def input_reorganize(self):
        d = self.X.shape[1]
        x1 = self.X[:, 0:15]
        x2 = self.X[:, 15:d]
        x2_mean = np.reshape(x2, [x2.shape[0], int(d / 46), 46])
        #x1 = np.concatenate([x1, np.nanmean(x2_mean, axis=2), x2], axis=1)
        x1 = np.concatenate([x1, x2_mean[:, :, 45], x2], axis=1)
        # x2= np.transpose(x2,(0,2,1))
        self.X = x1


def test_set_GPU(xtest, ytest):
    xtest = xtest.astype(np.float32)
    ytest = ytest.astype(np.float32)
    xtest = torch.from_numpy(xtest).to(device)
    ytest = torch.from_numpy(ytest).to(device)
    ytest = ytest.reshape((ytest.shape[0], 1))
    return [xtest, ytest]


def get_9km_of_a_network(input_fine,input_9km,network_name):
    temp = input_fine[input_fine['network'].isin(network_name)]
    temp = temp.groupby(['r', 'c']).size().reset_index(name='Freq')
    input_9km=input_9km[input_9km['r'].isin(list(temp['r']))]
    input_9km=input_9km[input_9km['c'].isin(list(temp['c']))]
    return input_9km.rename(columns={'SMAP':'gt'})