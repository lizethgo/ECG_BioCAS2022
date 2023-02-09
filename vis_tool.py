# -*- coding: utf-8 -*-
"""
Created on Fri May  7 20:36:52 2021
Author       : Lizeth Gonzalez Carabarin
Organization : Eindhoven University of Technology (TU/e)
Description  : Visualization tools for debugging

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re




def fill_label(label, sample, size):
    label_full = np.zeros(size)
    for i in range(1,len(sample)):
        
        label_full[int(sample[i-1]):int(sample[i])] = label[i-1]
    return label_full


"""
Information of datatset_1
Name: MIT-BIH
The recordings were digitized at 360 samples per second per channel with 11-bit resolution
files: 48
labels: ['0', '0\t(AB', '0\t(AFIB', '0\t(AFL', '0\t(B', '0\t(BII’, '0\t(IVR', '0\t(N', '0\t(NOD', '0\t(P', '0\t(PREX', '0\t(SVTA',
'0\t(T', '0\t(VFL', '0\t(VT', '0\tMISSB', '0\tPSE', '0\tTS']

    
== '0\t(VT' -> 1 shockable
=! '0\t(VT' -> 0 non-shockable
    
"""

### reading first file to get init information  
col_names =['Time', 'Sample #', 'Type', 'Sub', 'Chan', 'Num\tAux']
dataset_1 = pd.read_csv('./mit_bih/100.csv')
test_ = dataset_1.values
x1 = dataset_1.iloc[:,-1].values ## channel 1
annot = pd.read_csv('./mit_bih/100annotations.txt', sep=r'\s{2,}', engine='python', names=col_names, header=1)

## subsampling
x1_sub = np.delete(x1, np.arange(0, x1.size, 4))
x1_sub = np.delete(x1_sub, np.arange(0, x1_sub.size, 13))


### getting lengths 
size = np.shape(x1)[0]  ## original length of each file
## size = 650000, corresponding to 1805.55 seconds or ~30 mins (360 samples per second)
size_sub = np.shape(x1_sub)[0] ## subsampled length 
## size = 450000, corresponding to 1800.00 seconds or 30 mins (250 samples per second)
size_annot = np.shape(annot)[0]
n_files = 48  ## number of total files

## selection of windows time
time_win = 3 #(sec.)

file = 0
size_label = 0
annot_shape_ = 0
size_prev = 0


### getting size for labels
for i in range(100,234+1):
    try:
        annot_ = pd.read_csv('./mit_bih/{}annotations.txt'.format(i), sep=r'\s{2,}', engine='python', names=col_names, header=1)
        size_label = np.shape(annot_)[0] + size_label
    except FileNotFoundError:
        print('Warning: No file found: {}'.format(i))
        
### generating empty arrays
x1_sub_mit = np.zeros((size_sub*n_files))  ## subsampled channel 1 (250 Hz)
x2_sub_mit = np.zeros((size_sub*n_files)) ## subsampled channel 2 (250 Hz)

#sample = np.zeros(size_label)
      
            
for i in range(100,234+1):
    try:
        dataset_1 = pd.read_csv('./mit_bih/{}.csv'.format(i))
        test_ = dataset_1.values
        x1 = dataset_1.iloc[:,-1].values ## channel 1
        x2 = dataset_1.iloc[:,-2].values ## channel 2

        
        ### subsampling to match frequency resolution of the other datasets
        file = file + 1
        x1_sub_ = np.delete(x1, np.arange(0, x1.size, 4))
        x1_sub_mit[(file-1)*size_sub:(file-1)*size_sub+size_sub] = np.delete(x1_sub_, np.arange(0, x1_sub_.size, 13))
#
        x2_sub_ = np.delete(x2, np.arange(0, x1.size, 4))
        x2_sub_mit[(file-1)*size_sub:(file-1)*size_sub+size_sub] = np.delete(x2_sub_, np.arange(0, x2_sub_.size, 13))    
        
        annot_ = pd.read_csv('./mit_bih/{}annotations.txt'.format(i), sep=r'\s{2,}', engine='python', names=col_names, header=1)
        ## annot contains the medical specialists annotations. We are interested in the 'aux' column, which provides the different 
        ## patterns for arrythmia
            
        u, index = np.unique(annot_.values[:,5],return_index=True)
        print(u)
        for k in range(len(index)):
            print(annot_.values[index[k],1])


        if i == 100:
            label = annot_.values[:,5].astype(str)
            sample = annot_.values[:,1].astype(int)

        else:
            label = np.concatenate((label, annot_.values[:,5].astype(str)))
            sample = np.concatenate((sample, annot_.values[:,1].astype(int)+size_prev))
            
        #label[annot_shape_:annot_shape_+np.shape(annot_)[0]] = annot_.values[:,2]
        #label[annot_shape_:annot_shape_+np.shape(annot_)[0]] = get_label(annot_.values[:,2])
        
        annot_shape_ = np.shape(annot_)[0] + annot_shape_
        size_prev = file*size
    
    except FileNotFoundError:
        print('Warning: No file found: {}'.format(i))
    

#plt.plot(x1[0:1000])
#plt.show()NP.
        

def get_label_mit(annot_label):
    annot_label[annot_label!='0\t(VT']='0'
    annot_label[annot_label=='0\t(VT']='1'
    label = annot_label.astype(int)
    
    return label

label_mit = get_label_mit(label)    
sample_sub_mit =  450000 *(sample / 650000)

size_ = np.shape(x1_sub_mit)[0]        
label_full_sub_mitbih = fill_label(label_mit, sample_sub_mit, size_) 


file = 100
sample_init = 0
length = 1000  
    
dataset_1 = pd.read_csv('./mit_bih/100.csv')
test_ = dataset_1.values
x1 = dataset_1.iloc[:,-1].values ## channel 1
annot = pd.read_csv('./mit_bih/100annotations.txt', sep=r'\s{2,}', engine='python', names=col_names, header=1)
    
