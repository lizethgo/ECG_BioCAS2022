# -*- coding: utf-8 -*-
"""
Created on Fri May  7 20:36:52 2021

@author: 20195088
"""

from __future__ import division

#### general libraries
import tensorflow as tf
import os
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
# Currently, memory growth needs to be the same across GPUs
   try:
       for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
   except RuntimeError as e:
       print(e)

tf.config.experimental.set_visible_devices(gpus[3],'GPU')



def preprocessing(selector=1):

    
    def fill_label(label, sample, size):
        label_full = np.zeros(size)
        for i in range(1,len(sample)):
            
            label_full[int(sample[i-1]):int(sample[i])] = label[i-1]
        return label_full
    
    
    if selector == 1:
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
                x1 = dataset_1.iloc[:,-2].values ## channel 1
                x2 = dataset_1.iloc[:,-1].values ## channel 2
        
                
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
    
    
    if selector == 2:
        """
        Information of datatset_2
        Name: CUDB
        https://physionet.org/content/cudb/1.0.0/ 
        files: 35
        sampling frequency: 250 Hz
        channels: 1
        record time: ~8.5 min
        annotation.__dict__['aux_note']: array(['', '(AF', '(N', '(VF', '(VT'], dtype='<U4')
        """
        
        import wfdb
        
        
        files = 35
        
        
        #record1 = wfdb.rdheader('./CUDB/cu01')
        record2 = wfdb.rdrecord('./CUDB/cu01') 
        #display(record2.__dict__)
        x=record2.__dict__['p_signal'][:,0]
        # size
        x_cudb = np.zeros(np.shape(x)[0]*files)
        size_prev = 0
        
        for i in range(1,36):
            
        
            record1 = wfdb.rdrecord('./CUDB/cu{:02d}'.format(i)) 
            #display(record2.__dict__)
            x_cudb[np.shape(x)[0]*(i-1):np.shape(x)[0]*(i-1)+np.shape(x)[0]]=record1.__dict__['p_signal'][:,0]
            
            
            annotation = wfdb.rdann('./CUDB/cu{:02d}'.format(i), 'atr')
            #display(annotation.__dict__)
            
            if i == 1:
                sample_cudb = annotation.__dict__['sample']
                label_cudb = annotation.__dict__['aux_note']
            else:
                sample_cudb = np.concatenate((sample_cudb, annotation.__dict__['sample']+size_prev))
                label_cudb = np.concatenate((label_cudb,annotation.__dict__['aux_note']))
                
            size_prev = i*np.shape(x)[0] ### This will be added to next sample_cudb in order to shift all samples
            #print(size_prev)
                
        
        
        def get_label_cudb(annot_label):
        
            annot_label[annot_label=='(VT']='1'
            annot_label[annot_label=='(VF']='1'
            annot_label[annot_label!='1']='0'
            
            label = annot_label.astype(int)
            
            return label
        
        
        label_cudb = get_label_cudb(label_cudb)
        
        
        size = np.shape(x_cudb)[0]        
        label_full_cudb = fill_label(label_cudb, sample_cudb, size)        
        
        
        
    if selector == 3:
        
        """
        Information of datatset_2
        Name: 
        files: 22
        https://physionet.org/content/vfdb/1.0.0/
        sampling frequency: 250 Hz
        channles: 2
        time: 35 min recording
        annotation.__dict__['aux_note']:
        array(['(AFIB', '(ASYS', '(B', '(BI', '(HGEA', '(N', '(NOD', '(NOISE',
               '(NSR', '(PM', '(SBR', '(SVTA', '(VER', '(VF', '(VFIB', '(VFL',
               '(VT'], dtype='<U7')
            
        """
        
        files = 22
        #record1 = wfdb.rdheader('./CUDB/cu01')
        record2 = wfdb.rdrecord('./mit_bih_m/419') 
        #display(record2.__dict__)
        x=record2.__dict__['p_signal']
        # size
        #x_mitbih_m_1 = np.zeros(np.shape(x)[0]*files)
        #x_mitbih_m_2 = np.zeros(np.shape(x)[0]*files)
        
        x_mitbih_m_1 = np.zeros(2)
        x_mitbih_m_2 = np.zeros(2)
        #init var
        file = 0
        n =  0 # files with noise
        size_previous_signal = 0 # this variable is used as an accumulator to shift the values of all samples
        sample_noiseless_mitbih_m = np.zeros(2) # contains the final noiseless samples 
        label_noiseless_mitbih_m = np.zeros(2) # contains the final noiseless labels 
        
        for i in range(418,500):
            try:
                record1 = wfdb.rdrecord('./mit_bih_m/{}'.format(i)) 
                file = file + 1
                
                sample_noise_mitbih_m_init = []
                sample_noise_mitbih_m_end = []
                indexes_noise_end = []
                
                dif = 0
        
                
                #display(record2.__dict__)
                        #### cleaning signal (eliminating NOISE samples):
                annotation = wfdb.rdann('./mit_bih_m/{}'.format(i), 'atr')
                label_mitbih_m = np.array(annotation.__dict__['aux_note'])
                sample_mitbih_m = np.array(annotation.__dict__['sample'])
                noise = ('(NOISE' in label_mitbih_m)
                noise_t = np.where(label_mitbih_m=='(NOISE')
                print(noise)
        
                
                if noise:
                    print('file: {} is modified beacuse Noise corruption'.format(i))
                    
                    #print(label_mitbih_m)
                    print(sample_mitbih_m[-1])
                    
                    ### getting the sequences of noise init and end to further remove those samples
                    for j in range(len(sample_mitbih_m)):
                        if (label_mitbih_m[j]=='(NOISE'):
                            #print('FIRST NOISE 1 at ', j)
                            if j > 0:
                                if (label_mitbih_m[j-1] != '(NOISE'): # First sequence with noise
                                    #print('FIRST NOISE')
                                    sample_noise_mitbih_m_init.append(sample_mitbih_m[j]) 
                                    
                                    if j == (len(sample_mitbih_m)-1): ## in case last sample is NOISE
                                        sample_noise_mitbih_m_end.append(np.shape(x)[0])
                                        #indexes_noise_end.append(j)
                                    
        
                        elif ((label_mitbih_m[j]!='(NOISE') and (label_mitbih_m[j-1]=='(NOISE') and (j!=0)):
                            sample_noise_mitbih_m_end.append(sample_mitbih_m[j])
                            indexes_noise_end.append(j)
                            
        
                            
                    
                    ## removing noisy samples
                    #print(sample_noise_mitbih_m_init)
                    #print(sample_noise_mitbih_m_end)
                    #x_mitbih_m_1 = record1.__dict__['p_signal'][:,0]
                    #print('SHAPE IS: ', np.shape(x_mitbih_m_1))
                    #x_mitbih_m_1 = np.zeros((2))
                    for j in range(len(sample_noise_mitbih_m_init)):
                        #print(j,x)
                        if j == 0:
                            
                            """check the first noisy sample and get from 0 to sample_noise_mitbih_m_init
                            """
                            s = int(np.where(sample_mitbih_m==sample_noise_mitbih_m_init[j])[0])   ## position of the noisy sample in sample_mitbih_m
                            #x_mitbih_m_1 = np.concatenate((x_mitbih_m_1, record1.__dict__['p_signal'][int(sample_mitbih_m[s-1]):int(sample_noise_mitbih_m_init[j]),0]))
                            x_mitbih_m_1 = np.concatenate((x_mitbih_m_1, record1.__dict__['p_signal'][0:int(sample_noise_mitbih_m_init[j]),0]))
                            x_mitbih_m_2 = np.concatenate((x_mitbih_m_2, record1.__dict__['p_signal'][0:int(sample_noise_mitbih_m_init[j]),1]))
        
                            
                            #print(j, '', np.shape(record1.__dict__['p_signal'][0:int(sample_noise_mitbih_m_init[j]),0]))
                        elif j == (len(sample_noise_mitbih_m_init)-1):
                            
                            """check the kast noisy sample. There are two possibilities: a) the last sample is NOT NOISE, then we take sample_noise_mitbih_m_end[-1]) to the last sample;
                            b) the last sample is NOISE, for which we just take from sample_noise_mitbih_m_end[j-1] to sample_noise_mitbih_m_init[j], in this case int(sample_noise_mitbih_m_end[-1]):-1
                            will be automatically 0.
                            """
                            x_mitbih_m_1 = np.concatenate((x_mitbih_m_1, record1.__dict__['p_signal'][int(sample_noise_mitbih_m_end[j-1]):int(sample_noise_mitbih_m_init[j]),0]))
                            x_mitbih_m_2 = np.concatenate((x_mitbih_m_2, record1.__dict__['p_signal'][int(sample_noise_mitbih_m_end[j-1]):int(sample_noise_mitbih_m_init[j]),1]))
                            #print(j, '', np.shape(record1.__dict__['p_signal'][int(sample_noise_mitbih_m_end[j-1]):int(sample_noise_mitbih_m_init[j]),0]))
                            x_mitbih_m_1 = np.concatenate((x_mitbih_m_1, record1.__dict__['p_signal'][int(sample_noise_mitbih_m_end[-1]):-1,0]))
                            x_mitbih_m_2 = np.concatenate((x_mitbih_m_2, record1.__dict__['p_signal'][int(sample_noise_mitbih_m_end[-1]):-1,1]))
                            #print(j, '', np.shape(record1.__dict__['p_signal'][int(sample_noise_mitbih_m_end[-1]):-1,0]))
                            #label_test = np.delete(label_mitbih_m,s)
                            #sample_test = np.delete(sample_mitbih_m,s)
                        else:
                            """check the intermediate noisy samples and get from the last noisy sample sample_noise_mitbih_m_end[j-1] to the init of next one sample_noise_mitbih_m_init[j]
                            """
                            
                            x_mitbih_m_1 = np.concatenate((x_mitbih_m_1, record1.__dict__['p_signal'][int(sample_noise_mitbih_m_end[j-1]):int(sample_noise_mitbih_m_init[j]),0]))
                            x_mitbih_m_2 = np.concatenate((x_mitbih_m_2, record1.__dict__['p_signal'][int(sample_noise_mitbih_m_end[j-1]):int(sample_noise_mitbih_m_init[j]),1]))
                            #print(j, '', np.shape(record1.__dict__['p_signal'][int(sample_noise_mitbih_m_end[j-1]):int(sample_noise_mitbih_m_init[j]),0]))
                            #label_test = np.delete(label_mitbih_m,s)
                            #sample_test = np.delete(sample_mitbih_m,s)
        
                    
                    #print('SHAPE IS: ', np.shape(x_mitbih_m_1))
                    ## eliminating noisy lables / and samples
                    indexes_noise_init = np.where(label_mitbih_m=='(NOISE') # indexes of 'NOISE' labels, which are also the indexed for their samples
                    
                    """ delete NOISE labels"""
                    label_noiseless = np.delete(label_mitbih_m, indexes_noise_init)
                    label_noiseless_mitbih_m = np.concatenate((label_noiseless_mitbih_m, label_noiseless))
                    
                    sample_mitbih_m_test = np.copy(sample_mitbih_m)
                    #sample_noiseless = np.delete(sample_mitbih_m, indexes_noise_init)
                    
                    for k in range(len(indexes_noise_end)):
                        
                        dif = (sample_noise_mitbih_m_end[k]-sample_noise_mitbih_m_init[k])+dif 
                        
                        if np.shape(indexes_noise_end)[0] == np.shape(indexes_noise_init)[1]:   ### in case NOISE is no the last sample
                            
                            sample_mitbih_m_test[int(indexes_noise_init[0][0+k]):-1]=sample_mitbih_m[int(indexes_noise_init[0][0+k]):-1]-dif
                            
                        else: ### NOISE is the last sample
                            """ the difference must be accumulative so all samples are appropiatly shifted  """
            
                            sample_mitbih_m_test[int(indexes_noise_init[0][0+k]):int(indexes_noise_init[0][k+1])]=sample_mitbih_m[int(indexes_noise_init[0][0+k]):int(indexes_noise_init[0][k+1])]-dif
                        
                    """ at the end just removed the samples that marked the beginning of NOISE"""
                    sample_noiseless = np.delete(sample_mitbih_m_test, indexes_noise_init)
                    print('SAMPLE NOISELESS', sample_noiseless[-1])
                    sample_noiseless = sample_noiseless + size_previous_signal
                    sample_noiseless_mitbih_m = np.concatenate((sample_noiseless_mitbih_m, sample_noiseless))
                    print('CUMULATIVE SAMPLE NOISELESS', sample_noiseless_mitbih_m[-1])
                        
                    
                    
                    
                    
        
                else:
                    #x_mitbih_m_1[np.shape(x)[0]*(file-1):np.shape(x)[0]*(file-1)+np.shape(x)[0]]=record1.__dict__['p_signal'][:,0]
                    #x_mitbih_m_2[np.shape(x)[0]*(file-1):np.shape(x)[0]*(file-1)+np.shape(x)[0]]=record1.__dict__['p_signal'][:,1]
                        
                    if i == 418:
                        sample_noiseless_mitbih_m = annotation.__dict__['sample']
                        label_noiseless_mitbih_m = annotation.__dict__['aux_note']
                        x_mitbih_m_1 = record1.__dict__['p_signal'][:,0]
                        x_mitbih_m_2 = record1.__dict__['p_signal'][:,1]
                    else:
                        sample_noiseless_mitbih_m = np.concatenate((sample_noiseless_mitbih_m, annotation.__dict__['sample']))
                        label_noiseless_mitbih_m = np.concatenate((label_noiseless_mitbih_m, annotation.__dict__['aux_note']))
                        x_mitbih_m_1 = np.concatenate((x_mitbih_m_1, record1.__dict__['p_signal'][:,0]))
                        x_mitbih_m_2 = np.concatenate((x_mitbih_m_2, record1.__dict__['p_signal'][:,1]))
                        
                """ This is necessary in order to shift further samples"""        
                size_previous_signal = np.shape(x_mitbih_m_1)[0]
                print('SIZE IS', size_previous_signal)
                
                        
                    
                
                
        
                #print(np.shape(sample_mitbih_m))
                #display(annotation.__dict__['aux_note'][0])
                #display(annotation.__dict__['aux_note'][-1])
                
        
        
            except FileNotFoundError:
                print('Warning: No file found: {}'.format(i))
                
        ### deleting initial zeros that serve to start concatenation         
        x_mitbih_m_1 = np.delete(x_mitbih_m_1, (0,1))
        x_mitbih_m_2 = np.delete(x_mitbih_m_2, (0,1))
        
        sample_noiseless_mitbih_m = np.delete(sample_noiseless_mitbih_m, (0,1))
        label_noiseless_mitbih_m = np.delete(label_noiseless_mitbih_m, (0,1))
        
        def get_label_mitbih_m(annot_label):
        
            annot_label[annot_label=='(VT']  ='1'
            annot_label[annot_label=='(VF']  ='1'
            annot_label[annot_label=='(VFIB']='1'
            annot_label[annot_label!='1']    ='0'
            
            label = annot_label.astype(int)
            
            return label
        
        label_noiseless_mitbih_m = get_label_mitbih_m(label_noiseless_mitbih_m)
        
        size = np.shape(x_mitbih_m_1)[0]        
        label_full_mitbih_m = fill_label(label_noiseless_mitbih_m, sample_noiseless_mitbih_m, size)  
        
    
    
        
    """
    *******************************************************************************
    dataset               x_variables                   y_variables
    *******************************************************************************
    mitbih     - x: x1_sub_mit, x1_sub_mit     y: label_mit, sample_sub_mit 
    cudb       - x: x_cudb                     y: label_cudb, sample_cudb 
    mithbih-m: - x: x_mitbih_m_1, x_mitbih_m_2 y: label_mitbih_m, sample_mitbih_m
    *******************************************************************************
    """
    
    window_time = 2
    sampling_freq = 250
    length = int(window_time*sampling_freq)
    channels = 1 # select number of channels
    
    
    """
    selector specify:
    0:  mitbih
    1:  cudb
    2:  mithbih-m
    3:  all
    
    batches = # of batches generated after diving the full data into window_time
    """
    
    if selector == 1:
        x_ = np.zeros((np.shape(x1_sub_mit)[0],channels))
        x_ = x1_sub_mit
        #y_samples = sample_sub_mit 
        y_label = label_full_sub_mitbih
        batches=int(np.shape(x_)[0]/length)
    elif selector == 2:
        x_ = np.zeros((np.shape(x_cudb)[0],channels))
        x_ = x_cudb
        #y_samples = sample_cudb
        y_label = label_full_cudb
        batches=int(np.shape(x_)[0]/length)
    elif selector == 3:
        x_ = np.zeros((np.shape(x_mitbih_m_1)[0],channels))
        x_ = x_mitbih_m_1
        #y_samples = sample_mitbih_m
        y_label = label_full_mitbih_m
        batches=int(np.shape(x_)[0]/length)
    else:
        x_ = np.zeros((np.shape(x1_sub_mit)[0]+np.shape(x_cudb)[0]+np.shape(x_mitbih_m_1)[0],channels))
        x_ = np.concatenate((x1_sub_mit,x_cudb,x_mitbih_m_1))
        y_label = np.concatenate((label_full_sub_mitbih, label_full_cudb + np.shape(x1_sub_mit)[0], label_full_mitbih_m + np.shape(x_mitbih_m_1)[0])) 
        #y_samples = np.concatenate((sample_sub_mit, sample_cudb + np.shape(x1_sub_mit)[0], sample_mitbih_m + np.shape(x_mitbih_m_1)[0])) 
        x_ = np.expand_dims(x_, axis=1)
        batches=int(np.shape(x_)[0]/length)
    
    x_batches = np.zeros((batches,length))
    y_batches = np.zeros((batches))
    
    
    """
    size of dataset:
    x_batches[batches,length]
    """
    # Organizing data (x,y) into batches 
    for i in range(0,batches):
        #accu = i*length   ## cumulative variable
        x_batches[i,:] = x_[i*length:i*length+length]
        if np.sum(y_label[i*length:i*length+length])>0:
             y_batches[i] = 1
        else:  ## if there is at least one, label y_batches as 1
             y_batches[i] = 0
            
    
    
    
    """
    synthetic data
    """
    rep = 100 ## number of times data is generated for each label = 1
    index_vt = np.where(y_batches==1)  # indexes for labe = 1
    
    
    x_batches_ = np.zeros((1,length))
    #print(np.shape(x_batches_))
    print('SYNTHETIC BATCHES', rep*np.size(index_vt))
    print('ORIGINAL NUMBER OF BATCHES', np.shape(x_batches)[0])
    
    for j in range(np.size(index_vt)):
        data = x_batches[index_vt[0][j],:]
        data = np.reshape(data, (1,np.size(data)))
        #plt.plot(data[0,:]) ## visualizing VT data
        #plt.show()
        x_batches_ = np.concatenate((x_batches_, data), axis=0)
        #print(np.shape(x_batches_))
        #y_batches_ = np.concatenate((y_batches_, np.array([1])), axis=0)
    
    x_batches_ = np.delete(x_batches_, 0, axis=0)
    
    x_batches_ = np.repeat(x_batches_, rep, axis=0)   ### repeat rep times data with label = 1
    y_batches_ = np.ones(int(rep*np.size(index_vt))) 
    
    ### final x_train / y_train with synthetic data
    #x_batches = np.concatenate((x_batches, x_batches_))
    #y_batches = np.concatenate((y_batches, y_batches_))
    
    print('FINAL NUMBER OF BATCHES', np.shape(x_batches)[0])    
    
    
    
    
    from sklearn.utils import shuffle
    data_x, data_y = shuffle(x_batches, y_batches)
    
    """
    By randomnly shuffling the batches, and then dividing them 0.75 training 0.25 testing I ensure each partition has the
    same proportion of ones, this is both partitions will have the same proportion of non-seizure / seizure data
    """
    train_ratio = 0.75
    train_samples = int(np.shape(data_y)[0]*train_ratio)
    test_samples = np.shape(data_y)[0] - train_samples
    
    data_y_train = data_y[0:train_samples]
    data_y_test = data_y[train_samples:]
    
    data_x_train = data_x[0:train_samples]
    data_x_test = data_x[train_samples:]
    
    data_x_train = np.concatenate((data_x_train, x_batches_))
    data_y_train = np.concatenate((data_y_train, y_batches_))

    from sklearn.utils import shuffle
    data_x_train, data_y_train = shuffle(data_x_train, data_y_train)
    
    return data_x_train, data_x_test, data_y_train, data_y_test


#data_x_train, data_x_test, data_y_train, data_y_test = preprocessing(selector=1)



import scipy.io
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import signal



def data_to_stft(x, fs, n, n_ch):
    f = []
    t = []
    n_freq = 6
    print(np.shape(x))
    t1,t2,t3 = signal.stft(x[0,0,:,0], fs, nperseg=n)
    z = np.zeros(shape=(np.shape(x)[0],n_freq,np.shape(t3)[1], n_ch))
    for i in range(np.shape(x)[0]):
        #print(np.shape(x[i,j,:,0]))
        f_,t_,z_ = signal.stft(x[i,0,:,0], fs, nperseg=n)
        #print(np.shape(f_))
        f.append(f_)
        t.append(t_)
        z[i,:,:,0] = z_[0:n_freq,:] 
    return f,t,z


    
    

    
#hours = [5,6,8,12] ### These are the files to be included, which should be avilable in the same folder as this file is stored
#hours = [121, 287]
#p = '04' ## patient number
#w_size = 1024 # modify this value according to the desired window size

x_train_, x_test_, y_train_, y_test_ = preprocessing(selector=1)

## expandind to 3D to match input size of Conv2D
x_train_ = np.expand_dims(x_train_, axis=-1)
x_train_ = np.expand_dims(x_train_, axis=1)
x_test_ = np.expand_dims(x_test_, axis=-1)
x_test_ = np.expand_dims(x_test_, axis=1)
print(np.shape(x_train_))
print(np.shape(x_test_))

"""
optinal: depending on the classification type and activation function, y_test and y_train
could be eaither binary or categorical
"""

from tensorflow.keras.utils import to_categorical
y_test_ = to_categorical(y_test_)
y_train_ = to_categorical(y_train_)


"""
Data normalization to [0,1]

"""
#x_train = (x_train_ + abs(np.min(x_train_)))  / (abs(np.min(x_train_))+np.max(x_train_)) 
#x_test = (x_test_ + abs(np.min(x_train_)))  / (abs(np.min(x_test_))+np.max(x_train_)) 



amp = 2 * np.sqrt(2)
f_train,t_train,x_train_images = data_to_stft(x_train_, 250, n=32, n_ch=1)

#f,t,z
f_test,t_test,x_test_images = data_to_stft(x_test_, 250, n=32, n_ch=1)
    

# for i in range(20):
    # plt.pcolormesh(t_train[i][:], f_train[i][0:6], np.abs(x_train_images[i,0:6,:,0]), vmin=0, vmax=np.max(np.abs(x_train_images)), shading='gouraud')
    # plt.title('STFT Magnitude')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
#


x_train_images = (x_train_images-np.mean(x_train_images))/(np.std(x_train_images)+1e-7) 
x_test_images = (x_test_images-np.mean(x_test_images))/(np.std(x_test_images)+1e-7) 


#
#
#for i in range(np.shape(x_train)[0]):
#    for j in range(n_ch):
#        x_train_images[i,j:j+1,:,:] = z[i*j:i*j+1][:][:]

### transforming to RGB


"""
Model definition: 2Conv2D+1FC
"""

#### importing layers from TF
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.activations import relu


N_in_length = np.shape(x_train_images)[-2]
N_in_height = np.shape(x_train_images)[-3]
N_out = np.shape(y_train_)[-1]
#
# x_ = Input(shape=(N_in_height, N_in_length, np.shape(x_train_images)[-1]))
# x = Conv2D(32,(3,3), activation=None , padding='same')(x_)
# x = BatchNormalization()(x)
# x = relu(x)

# x = Conv2D(64,(3,3), activation=None , padding='same')(x)
# x = BatchNormalization()(x)
# x = relu(x)

# #x = MaxPooling2D((2,2))(x)
# x = Conv2D(128,(3,3), activation=None , padding='same')(x)
# x = BatchNormalization()(x)
# x = relu(x)

# #x = MaxPooling2D((2,2))(x)
# x = Conv2D(256,(3,3), activation=None , padding='same')(x)
# x = BatchNormalization()(x)
# x = relu(x)


# x = MaxPooling2D((2,2))(x)
# x = Flatten()(x)
# x = Dropout(0.4)(x)

# x = Dense(512, activation=None)(x)
# x = BatchNormalization()(x)
# x = relu(x)
# x = Dropout(0.4)(x)

# y = Dense(2, activation='softmax')(x)


x_ = Input(shape=(N_in_height, N_in_length, np.shape(x_train_images)[-1]))
x = Conv2D(4,(3,1), activation=None , padding='same')(x_)
x = BatchNormalization()(x)
x = relu(x)

x = Conv2D(4,(3,1), activation=None , padding='same')(x)
x = BatchNormalization()(x)
x = relu(x)

x = MaxPooling2D((2,2))(x)

#x = MaxPooling2D((2,2))(x)
x = Conv2D(4,(3,1), activation=None , padding='same')(x)
x = BatchNormalization()(x)
x = relu(x)

#x = MaxPooling2D((2,2))(x)
x = Conv2D(4,(3,1), activation=None , padding='same')(x)
x = BatchNormalization()(x)
x = relu(x)

x = MaxPooling2D((3,2))(x)
x = Flatten()(x)
#x = Dropout(0.4)(x)

x = Dense(25, activation=None)(x)
x = BatchNormalization()(x)
x = relu(x)
x = Dropout(0.7)(x)

y = Dense(2, activation='softmax')(x)

model = Model(inputs=x_, outputs=y)
model.summary()

from tensorflow.keras.optimizers import Adam, SGD
#optimizer = Adam(learning_rate = 0.001)
optimizer = SGD(learning_rate = 0.001, momentum=0.9)

# callback_model_checkpoint=tf.keras.callbacks.ModelCheckpoint(
  # filepath='weights.{epoch:02d}-{val_categorical_accuracy:.4f}.hdf5',
  # #filepath='weights.{epoch:02d}.hdf5',
  # monitor = 'val_categorical_accuracy',
  # verbose = 0,
  # save_best_only = True,
  # save_weights_only = False,
  # save_freq = 'epoch'
# )


model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(x=x_train_images, y=y_train_, batch_size = 32, epochs =20, verbose=1)

model.evaluate(x=x_test_images, y=y_test_, batch_size = 32)

y_pred = model.predict(x_test_images)

predicted_s = 0
total_s = 0
for i in range(np.shape(y_test_)[0]):
	if (y_test_[i,1] == 1):
		total_s = total_s + 1
		#print(y_pred[0,1])
		if ((round(y_pred[i,1]) == 1)):
			predicted_s = predicted_s + 1
            
print(total_s) 
print(predicted_s) # TP
TP = predicted_s
FN = total_s - predicted_s

predicted_s_ = 0
total_s_ = 0
for i in range(np.shape(y_test_)[0]):
	if (y_test_[i,0] == 1):
		total_s_ = total_s_ + 1
		#print(y_pred[0,1])
		if ((round(y_pred[i,0]) == 1)):
			predicted_s_ = predicted_s_ + 1

print(total_s_)
print(predicted_s_)

TN = predicted_s_
FP =  total_s_ - predicted_s_
acc = (TP+TN)/(TP+TN+FN+FP)
sen = TP/total_s
spec = TN/total_s_
recall = TP/(TP+FN)
prec = TP/(TP+FP)


F1 = 1/(((1/recall)+(1/prec))/2)

print('ACCURACY', acc )
print('SENSITIVITY',sen )
print('SPECIFICITY', spec )
print('PRECISION', prec)
print('RECALL', recall )
print('F1', F1)


    
   