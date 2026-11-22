import os
import random
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


class SWaTSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/SWaT_Dataset_Normal_v1.csv', header=0, decimal=',')
        data = data.values[:, 1:-1]

        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.channels = data.shape[1]

        test_data = pd.read_csv(data_path + '/SWaT_Dataset_Attack_v0.csv')
        test_data.columns = test_data.columns.str.strip()
        y = test_data['State'].to_numpy()
        labels = []
        for i in y:
            if i == 'Attack':
                labels.append(1)
            else:
                labels.append(0)
        labels = np.array(labels)

        test_data = test_data.values[:, 1:-1]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)
        self.train = data
        self.test_labels = labels.reshape(-1, 1)

        labels = self.test_labels

        total_points = labels.shape[0]
        anomamy_points = np.sum(labels == 1)

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print(f"Anomaly rate: {(anomamy_points / total_points):.4f}")

    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            x = np.float32(self.train[index:index + self.win_size])
            x = np.transpose(x, (1, 0))
            return x, np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            x = np.float32(self.test[index:index + self.win_size])
            x = np.transpose(x, (1, 0))

            window_labels = self.test_labels[index: index + self.win_size]
            #  Mark a window anomalous if any point is anomalous.
            label = 1 if np.sum(window_labels) > 0 else 0
            return x, label
        else:
            x = np.float32(self.train[index:index + self.win_size])
            x = np.transpose(x, (1, 0))

            window_labels = self.test_labels[0: self.win_size]
            label = 1 if np.sum(window_labels) > 0 else 0

            return  x, label

class PSMSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.channels = data.shape[1]

        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        labels = self.test_labels

        total_points = labels.shape[0]
        anomamy_points = np.sum(labels == 1)

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print(f"Anomaly rate: {(anomamy_points / total_points):.4f}")

    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            x = np.float32(self.train[index:index + self.win_size])
            x = np.transpose(x, (1, 0))
            return x, np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            x = np.float32(self.test[index:index + self.win_size])
            x = np.transpose(x, (1, 0))

            window_labels = self.test_labels[index: index + self.win_size]
            label = 1 if np.sum(window_labels) > 0 else 0
            return x, label
        else:
            x = np.float32(self.train[index:index + self.win_size])
            x = x.transpose(x, (1, 0))
            window_labels = self.test_labels[0: self.win_size]
            label = 1 if np.sum(window_labels) > 0 else 0
            return x, label


class MSLSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.channels = data.shape[1]
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")

        labels = self.test_labels

        total_points = labels.shape[0]
        anomamy_points = np.sum(labels == 1)
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print(f"Anomaly rate: {(anomamy_points / total_points):.4f}")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            x = np.float32(self.train[index:index + self.win_size])
            x = np.transpose(x, (1, 0))
            return x, np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            x = np.float32(self.test[index:index + self.win_size])
            x = np.transpose(x, (1, 0))

            window_labels = self.test_labels[index: index + self.win_size]
            label = 1 if np.sum(window_labels) > 0 else 0
            return  x, label
        else:
            x = np.float32(self.train[index:index + self.win_size])
            x = np.transpose(x, (1, 0))

            window_labels = self.test_labels[0: self.win_size]
            label = 1 if np.sum(window_labels) > 0 else 0
            return x, label

class SMAPSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        self.channels = data.shape[1]
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")

        labels = self.test_labels

        total_points = labels.shape[0]
        anomamy_points = np.sum(labels == 1)

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print(f"Anomaly rate: {(anomamy_points / total_points):.4f}")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            x = np.float32(self.train[index:index + self.win_size])
            x = np.transpose(x, (1, 0))
            return x, np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            x = np.float32(self.test[index:index + self.win_size])
            x = np.transpose(x, (1, 0))

            window_labels = self.test_labels[index: index + self.win_size]
            label = 1 if np.sum(window_labels) > 0 else 0

            return x, label
        else:
            x = np.float32(self.train[index:index + self.win_size])
            x = np.transpose(x, (1, 0))

            window_labels = self.test_labels[0: self.win_size]
            label = 1 if np.sum(window_labels) > 0 else 0

            return x, label

class SMDSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.channels = data.shape[1]
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

        labels = self.test_labels

        total_points = labels.shape[0]
        anomamy_points = np.sum(labels == 1)

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print(f"Anomaly rate: {(anomamy_points / total_points):.4f}")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            x = np.float32(self.train[index:index + self.win_size])
            x = np.transpose(x, (1, 0))
            return x, np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            x = np.float32(self.test[index:index + self.win_size])
            x = np.transpose(x, (1, 0))

            window_labels = self.test_labels[index: index + self.win_size]
            label = 1 if np.sum(window_labels) > 0 else 0

            return x, label
        else:
            x = np.float32(self.train[index:index + self.win_size])
            x = np.transpose(x, (1, 0))
            window_labels = self.test_labels[0: self.win_size]
            label = 1 if np.sum(window_labels) > 0 else 0
            return x, label

class WADISegLoader(Dataset):
    def __init__(self, root_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # train data does not have label column
        train_data = pd.read_csv(os.path.join(root_path, 'WADI_14days.csv'), sep=",", skiprows=4)
        test_data = pd.read_csv(os.path.join(root_path, 'WADI_attackdata.csv'), sep = ",")

        # Remove the blank space in the columns
        train_data.columns = [col.strip(' ') for col in train_data.columns]
        test_data.columns = [col.strip(' ') for col in test_data.columns]

        labels = []
        for index, row in test_data.iterrows():
            date_temp=row['Date']
            date_mask="%m/%d/%Y"
            date_obj=datetime.strptime(date_temp, date_mask)
            time_temp=row['Time']
            time_mask="%I:%M:%S.%f %p"
            time_obj=datetime.strptime(time_temp,time_mask)

            if date_obj==datetime.strptime('10/9/2017', '%m/%d/%Y'):
                if time_obj>=datetime.strptime('7:25:00.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('7:50:16.000 PM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue

            if date_obj==datetime.strptime('10/10/2017', '%m/%d/%Y'):
                if time_obj>=datetime.strptime('10:24:10.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('10:34:00.000 AM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj>=datetime.strptime('10:55:00.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('11:24:00.000 AM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj>=datetime.strptime('11:30:40.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('11:44:50.000 AM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj>=datetime.strptime('1:39:30.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('1:50:40.000 PM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj>=datetime.strptime('2:48:17.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('2:59:55.000 PM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj>=datetime.strptime('5:40:00.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('5:49:40.000 PM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj>=datetime.strptime('10:55:00.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('10:56:27.000 AM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue

            if date_obj==datetime.strptime('10/11/2017', '%m/%d/%Y'):
                if time_obj>=datetime.strptime('11:17:54.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('11:31:20.000 AM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj>=datetime.strptime('11:36:31.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('11:47:00.000 AM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj>=datetime.strptime('11:59:00.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('12:05:00.000 PM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj>=datetime.strptime('12:07:30.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('12:10:52.000 PM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj>=datetime.strptime('12:16:00.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('12:25:36.000 PM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue
                elif time_obj>=datetime.strptime('3:26:30.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('3:37:00.000 PM', '%I:%M:%S.%f %p'):
                    labels.append('Attack')
                    continue

            labels.append('Normal')

        # labels = test_data["Attack LABLE (1:No Attack, -1:Attack)"].apply(lambda x: 0 if x == 1 else 1).values
        # test_data = test_data.drop(['Attack LABLE (1:No Attack, -1:Attack)'], axis=1)
        test_data = test_data.drop(test_data.columns[[0,1,2,50,51,86,87]],axis=1) # Drop the empty and date/time columns
        labels = [ int(l!= 'Normal' ) for l in labels]

        # Removing columns that only contain time and nans (data missing from the csv file)
        train_nan_columns = {col for col in train_data.columns if train_data[col].isna().all()}
        test_nan_columns = {col for col in test_data.columns if test_data[col].isna().all()}
        common_nan_columns = list(train_nan_columns.intersection(test_nan_columns))

        train_data = train_data.drop(train_data.columns[[0,1,2,50,51,86,87]],axis=1) # Drop the empty and date/time columns
        # train_data = train_data.drop(['Row', 'Date', 'Time'] + common_nan_columns, axis=1)
        # test_data = test_data.drop(['Row', 'Date', 'Time'] + common_nan_columns, axis=1)

        # fill the missing values
        train_data = train_data.interpolate().bfill()
        test_data = test_data.interpolate().bfill()

        train_data = train_data.values
        test_data = test_data.values

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        self.channels = train_data.shape[1]
        test_data = self.scaler.transform(test_data)

        data_len = len(train_data)
        self.train = train_data[:int(data_len * 0.8)]
        self.val = train_data[int(data_len * 0.8):]
        self.test = test_data
        self.test_labels = labels

        labels = np.asarray(self.test_labels)

        total_points = labels.shape[0]
        anomamy_points = np.sum(labels == 1)

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print(f"Anomaly rate: {(anomamy_points / total_points):.4f}")

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            x = np.float32(self.train[index:index + self.win_size])
            x = np.transpose(x, (1, 0))
            return x, np.float32(np.zeros(self.win_size))
        elif self.mode == 'val':
            x = np.float32(self.val[index:index + self.win_size])
            x = np.transpose(x, (1, 0))
            return x, np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            x = np.float32(self.test[index:index + self.win_size])
            x = np.transpose(x, (1, 0))

            window_labels = self.test_labels[index: index + self.win_size]
            label = 1 if np.sum(window_labels) > 0 else 0

            return x, label
        else:
            x = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            x = np.transpose(x, (1, 0))

            window_labels = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
            label = 1 if np.sum(window_labels) > 0 else 0

            return x, label


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', val_ratio=0.2):
    """
    model : 'train' or 'test'
    """

    if dataset == 'SMD':
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif dataset == 'MSL':
        dataset = MSLSegLoader(data_path, win_size, step, mode)
    elif dataset == 'SMAP':
        dataset = SMAPSegLoader(data_path, win_size, step, mode)
    elif dataset == 'PSM':
        dataset = PSMSegLoader(data_path, win_size, step, mode)
    elif dataset == 'SWaT':
        dataset = SWaTSegLoader(data_path, win_size, step, mode)
    elif dataset == 'WADI':
        dataset = WADISegLoader(data_path, win_size, step, mode)

    shuffle = False

    if mode == 'train':
        shuffle = True

        dataset_len = int(len(dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))

        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)

        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(dataset, val_sub_indices)

        train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=shuffle)

        # k_use_len = int(train_use_len*0.1)
        # k_sub_indices = indices[:k_use_len]
        # k_subset = Subset(dataset, k_sub_indices)
        # k_loader = DataLoader(dataset=k_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        #
        # return train_loader, val_loader, k_loader

        return train_loader, val_loader, dataset.channels

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
