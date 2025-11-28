
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import os
import numpy as np
import pandas as pd

class Dataset_Sticker(Dataset):
    def __init__(self, 
                 country,
                 store,
                 data_path='dataset',
                 seq_size=None, # (seq_len, label_len, pred_len)
                 scale: bool=True,
                 train_ratio=0.7,
                 flag='train'):
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

        self.root_path = os.path.join(data_path, "stickers_dataset")
        self.country = country
        self.store_type = store
        self.train_ratio = train_ratio


        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        store_list = ("discount_stickers", "premium_sticker_mart", "stickers_for_less")
        if self.store_type == 'all':
            dfs = []
            for store in store_list:
                csv_path = os.path.join(self.root_path, self.country, store + ".csv")
                df = pd.read_csv(csv_path)
                df = df.rename(columns={col: f"{store}_{col}" for col in df.columns if col != "date"})
                dfs.append(df)
            df_raw = dfs[0]
            for df in dfs[1:]:
                df_raw = df_raw.merge(df, on="date", how="inner")
        elif self.store_type in store_list:
            csv_path = os.path.join(self.root_path, self.country, self.store_type + ".csv")
            df_raw = pd.read_csv(csv_path)
        else:
            raise ValueError(f"Invalid store type: {self.store_type}")
        
        max_len = len(df_raw)   # 2557
        self.max_len = max_len

        training_length = max_len - self.seq_len - self.pred_len
        train_end = int(training_length * self.train_ratio)

        border1s = [0,
                    train_end, 
                    train_end,
                    max_len - self.seq_len - self.forecast_len]
        border2s = [train_end,
                    training_length,
                    training_length,
                    max_len - self.seq_len - self.forecast_len]
        
        
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

        df_stamp = df_raw[['date']][border1:border2+self.seq_len+self.pred_len]

        df_stamp['year'] = df_stamp.apply(lambda row: int(row['date'].split('-')[0]), 1)
        df_stamp['month'] = df_stamp.apply(lambda row: int(row['date'].split('-')[1]), 1)
        df_stamp['day'] = df_stamp.apply(lambda row: int(row['date'].split('-')[2]), 1)
        data_stamp = df_stamp.drop(['date'], axis=1).values

        self.data_x = data[border1:border2 + self.seq_len]
        self.data_y = data[border1 + self.seq_len:border2 + self.seq_len + self.pred_len]
        self.data_stamp = data_stamp
        print(f"{self.flag} data length: {len(self.data_x) - self.seq_len + 1}")

    def __len__(self):
        if self.flag == 'forecast':
            return len(self.data_x) - self.seq_len + 1
        else:
            length = len(self.data_x) - self.seq_len + 1
            return length

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len  # no decoder use this model
        r_begin = s_end
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        seq_y = self.data_y[s_begin:s_begin + self.pred_len]
        seq_y_mark = self.data_stamp[r_begin:r_end]

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


def data_provider(data_path,
                  country,
                  store,
                  seq_len,
                  label_len,
                  pred_len,
                  forecast_len,
                  train_ratio=0.7,
                  batch_size=16,
                  num_workers=2,
                  drop_last=False,
                  scale=True,
                  flag='train'):

    flag = flag.lower()
    assert flag in ['train', 'val', 'test', 'forecast'], f"Invalid flag {flag}"
    shuffle = True if flag == 'train' else False

    dataset = Dataset_Sticker(data_path=data_path,
                              country=country,
                              store=store,
                              seq_size=[seq_len, label_len, pred_len, forecast_len],
                              scale=scale,
                              train_ratio=train_ratio,
                              flag=flag)
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=drop_last)
    
    return dataset, dataloader

