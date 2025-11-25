import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import os
import numpy as np
import pandas as pd

class Dataset_Weather(Dataset):
    def __init__(self, 
                 city,
                 data_path='dataset',
                 seq_size=None, # (seq_len, label_len, pred_len, forecast_len)
                 scale: bool=True,
                 flag='train',
                 is_global=False):
        self.flag = flag.lower()

        if seq_size == None:
            self.seq_len = 30 * 3
            self.label_len = 30
            self.pred_len = 30
            self.forecast_len = 30
        else:
            self.seq_len = seq_size[0]
            self.label_len = seq_size[1]
            self.pred_len = seq_size[2]
            self.forecast_len = seq_size[3]

        assert flag.lower() in ['train', 'val', 'test', 'forecast']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'forecast': 3}
        self.set_type = type_map[flag]
        self.scale = scale
        self.is_global = is_global

        if self.is_global:
            self.root_path = os.path.join(data_path, "weather_dataset_global")
        else:
            self.root_path = os.path.join(data_path, "weather_dataset_korea")

        self.city = city

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.city) + '_2020-2024' + '.csv')
        
        max_len = len(df_raw)
        self.max_len = max_len

        train_end = int(max_len * 0.7)
        val_end = int(max_len * 0.85)

        # border1s = [0,
        #             train_end, 
        #             val_end,
        #             max_len - self.seq_len - self.forecast_len]
        # border2s = [train_end,
        #             val_end,
        #             max_len,
        #             max_len - self.forecast_len]
        
        # For prediction
        border1s = [0,
                    0, 
                    0,
                    max_len - self.seq_len - self.forecast_len]
        border2s = [max_len - self.seq_len - self.pred_len,
                    max_len - self.seq_len - self.pred_len,
                    max_len - self.seq_len - self.pred_len,
                    max_len - self.forecast_len]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]  # data without date
        df_data = df_raw[cols_data]
        self.df_data = df_data.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_weath = df_raw[['time']][border1:border2]
        df_weath['year']   = df_weath['time'].dt.year
        df_weath['month']  = df_weath['time'].dt.month
        df_weath['day']    = df_weath['time'].dt.day
        df_weath['hour']   = df_weath['time'].dt.hour
        df_weath['minute'] = df_weath['time'].dt.minute
        df_weath['second'] = df_weath['time'].dt.second
        data_weath = df_weath.drop(['time'], axis=1).values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_weath = data_weath    

    def __len__(self):
        if self.flag == 'forecast':
            return len(self.data_x) - self.seq_len + 1
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len  # no decoder use this model
        r_begin = s_end
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_weath[s_begin:s_end]

        seq_y = self.data_y[r_begin:r_end]
        seq_y_mark = self.data_weath[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def get_whole_data_without_tail(self, tail_len=16, var_type=np.ndarray):
        data = self.df_data[:-tail_len]
        if isinstance(self.df_data, var_type):
            return data
        else:
            return torch.tensor(data)
        
    def get_tail_data(self, tail_len=16, var_type=np.ndarray):
        data = self.df_data[-tail_len:]
        if isinstance(self.df_data, var_type):
            return data
        else:
            return torch.tensor(data)


def data_provider_weather(data_path,
                  city,
                  seq_len,
                  is_global,
                  label_len = None,
                  pred_len= None,
                  forecast_len = None,
                  batch_size=16,
                  num_workers=2,
                  drop_last=False,
                  scale=True,
                  flag='train'):

    flag = flag.lower()
    assert flag in ['train', 'val', 'test', 'forecast'], f"Invalid flag {flag}"
    shuffle = True if flag == 'train' else False

    dataset = Dataset_Weather(data_path=data_path,
                              city=city,
                              seq_size=[seq_len, label_len, pred_len, forecast_len],
                              scale=scale,
                              flag=flag,
                              is_global=is_global
                              )
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=drop_last)
    
    return dataset, dataloader

