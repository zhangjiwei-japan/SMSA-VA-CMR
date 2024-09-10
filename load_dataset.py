import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import torchvision.transforms as transforms  
import pandas as pd 
import numpy as np 
import h5py
import os  

def load_data_visual_audio(load_path):
    f = h5py.File(load_path,'r')
    f.keys()
    lab_test = f["lab_test"]
    visual_test = f["visual_test"]
    audio_test = f["audio_test"]
    visual_train = f["visual_train"]
    audio_train = f["audio_train"]
    lab_train = f["lab_train"] 

    return visual_train,audio_train,lab_train,visual_test,audio_test,lab_test
    
def load_dataset_train(load_path,train_size):
    f = h5py.File(load_path,'r')
    f.keys()
    visual_train = f["visual_train"]
    audio_train = f["audio_train"]
    visual_test = f["visual_test"]
    audio_test = f["audio_test"]
    lab_train = f["lab_train"]
    Scaler_visual = MinMaxScaler().fit(visual_train) ##生成规则
    ##将规则应用于训练集
    visual_train_Scaler = Scaler_visual.transform(visual_train) 
    ##将规则应用于测试集
    visual_test_Scaler = Scaler_visual.transform(visual_test)

    Scaler_audio = MinMaxScaler().fit(audio_train) ##生成规则
    ##将规则应用于训练集
    audio_train_Scaler = Scaler_audio.transform(audio_train) 
    ##将规则应用于测试集
    audio_test_Scaler = Scaler_audio.transform(audio_test)
    # print(np.any(np.isnan(lab_train)))

    lab_train = torch.tensor(lab_train)
    lab_train = lab_train.view(lab_train.size(0))
    lab_train = lab_train.long()
    train_visual = TensorDataset(torch.tensor(visual_train_Scaler).float(), lab_train)
    train_audio = TensorDataset(torch.tensor(audio_train_Scaler).float(), lab_train)

    data_loader_visual = DataLoader(dataset=train_visual, batch_size=train_size, shuffle=False,drop_last=True)
    data_loader_audio = DataLoader(dataset=train_audio, batch_size=train_size, shuffle=False, drop_last=True)

    return data_loader_visual,data_loader_audio

def load_dataset_test(load_path,test_size):
    f = h5py.File(load_path,'r')
    f.keys()
    visual_train = f["visual_train"]
    audio_train = f["audio_train"]
    lab_test = f["lab_test"]
    visual_test = f["visual_test"]
    audio_test = f["audio_test"]
    
    Scaler_visual = MinMaxScaler().fit(visual_train) ##生成规则
    ##将规则应用于训练集
    visual_train_Scaler = Scaler_visual.transform(visual_train) 
    ##将规则应用于测试集
    visual_test_Scaler = Scaler_visual.transform(visual_test)
    
    Scaler_audio = MinMaxScaler().fit(audio_train) ##生成规则
    ##将规则应用于训练集
    audio_train_Scaler = Scaler_audio.transform(audio_train) 
    ##将规则应用于测试集
    audio_test_Scaler = Scaler_audio.transform(audio_test)
    # print(np.any(np.isnan(lab_train)))
    # print(len(lab_test))
    # print(np.any(np.isnan(visual_test)))
    lab_test = torch.tensor(np.array(lab_test))
    lab_test = lab_test.view(lab_test.size(0))
    lab_test = lab_test.long()
    test_visual = TensorDataset(torch.tensor(visual_test_Scaler).float(), lab_test)
    test_audio = TensorDataset(torch.tensor(audio_test_Scaler).float(), lab_test)

    data_loader_visual = DataLoader(dataset=test_visual, batch_size=test_size, shuffle=False)
    data_loader_audio = DataLoader(dataset=test_audio, batch_size=test_size, shuffle=False)

    return data_loader_visual,data_loader_audio

if __name__ == "__main__":
    load_path =  "dataset/vegas_feature.h5"
    # load_path =  "dataset/AVE_feature_updated_squence.h5"
    test_size = 128
    train_size = 128
    visual_test,audio_test = load_dataset_test(load_path,test_size)
    data_loader_visual,data_loader_audio = load_dataset_train(load_path,train_size)
    for epoch in range(2):
        for i, data in enumerate(zip(data_loader_visual, data_loader_audio)):
            inputs_visual = data[0][0].cuda()
            labels_visual = data[0][1].cuda()
            inputs_audio = data[1][0].cuda()
            labels_audio = data[1][1].cuda()
            
            print("epoch：", epoch, "的第" , i, "个inputs", inputs_visual.shape, "labels", labels_audio.shape)

