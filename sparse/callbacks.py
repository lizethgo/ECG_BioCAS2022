# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 08:37:49 2020

@author: rvsloun
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.activations import relu
from tensorflow.keras import regularizers

class sample_vis(tf.keras.callbacks.Callback):   

    def __init__(self, data, **kwargs):
        self.data = data

        super(sample_vis, self).__init__()   
    def on_train_begin(self, epoch, logs=None):
    
        points = 1000
    
        Fs=250.
        t = [i*1./Fs for i in range(200)]
        peak_frequency = np.zeros([points], dtype=np.int64)
        
        for i in range(0,points):
            output_0 = self.model.get_layer('conv1d').output
            intermediate_model_0=tf.keras.models.Model(inputs=self.model.input,outputs=output_0)
            #print(np.shape(x[i,:,:]))
            #filter_ = np.random.randint(1,32)
            filter_ = 1
            #print(np.shape(self.data))
            x = np.expand_dims(self.data[i,:,:], axis=0)
            #x = np.expand_dims(x, axis=-1)
            #print(np.shape(x))
            intermediate_prediction_0=intermediate_model_0.predict(x)
            #print(np.shape(intermediate_prediction_0))
            intermediate_prediction_0 = np.squeeze(intermediate_prediction_0[:,:,filter_])
            
            fourier = np.fft.fft(intermediate_prediction_0)
            frequencies = np.fft.fftfreq(len(t), 0.004)  # where 0.005 is the inter-sample time difference
            #positive_frequencies = frequencies[np.where(frequencies >= 0)] 
             
            #magnitudes = abs(fourier[np.where(frequencies >= 0)])  # magnitude spectrum
            magnitudes = abs(fourier[0:int(np.shape(frequencies)[0]/2)])  # magnitude spectrum
            #print(np.shape(magnitudes))  
            peak_frequency[i] = int(np.argmax(magnitudes)+1)
        #print(peak_frequency)
        #print(np.bincount(peak_frequency))
        np.savetxt('train_begin_0st.txt', peak_frequency, delimiter=', ')
        
        
        peak_frequency = np.zeros([points], dtype=np.int64)
        ## FFT of intermediate prediction
        for i in range(0,points):
            output_1 = self.model.get_layer('sparseconnect_cnn_layer_1d').output
            intermediate_model_1=tf.keras.models.Model(inputs=self.model.input,outputs=output_1)
            #print(np.shape(x[i,:,:]))
            #filter_ = np.random.randint(1,32)
            filter_ = 1
            x = np.expand_dims(self.data[i,:,:], axis=0)
            #x = np.expand_dims(x, axis=-1)
            #print(np.shape(x))
            intermediate_prediction_1=intermediate_model_1.predict(x)
            intermediate_prediction_1 = np.squeeze(intermediate_prediction_1[:,:,filter_], axis=0)
            fourier = np.fft.fft(intermediate_prediction_1)
            frequencies = np.fft.fftfreq(len(t), 0.004)  # where 0.005 is the inter-sample time difference
            #positive_frequencies = frequencies[np.where(frequencies >= 0)] 
            #print(np.shape(frequencies)) 
            #print(np.shape(fourier))             
            #magnitudes = abs(fourier[np.where(frequencies >= 0)])  # magnitude spectrum
            magnitudes = abs(fourier[0:int(np.shape(frequencies)[0]/2)])
            #print(np.shape(magnitudes))  
            peak_frequency[i] = int(np.argmax(magnitudes)+1)
        #print(peak_frequency)
        #print(np.bincount(peak_frequency))
        np.savetxt('train_begin_1st.txt', peak_frequency, delimiter=', ')
        #np.savetxt('train_begin_1sr_s.txt', peak_frequency, delimiter=', ')
        
            
        #print(type(peak_frequency))
        #print(type(np.bincount(peak_frequency)))        
            
        #size_=range(0,np.shape(np.bincount(peak_frequency))[0])
        #print(np.shape(size_))    
        #print(np.bincount(peak_frequency))    
        #plt.bar(size_,np.bincount(peak_frequency),width=1)
        #plt.show()
        
        ### second layer
        ## FFT of intermediate prediction
        Fs=125.
        peak_frequency = np.zeros([points], dtype=np.int64)
        for i in range(0,points):
            output_2 = self.model.get_layer('sparseconnect_cnn_layer_1d_1').output
            intermediate_model_2=tf.keras.models.Model(inputs=self.model.input,outputs=output_2)
            
            #print(np.shape(x[i,:,:]))
            #filter_ = np.random.randint(1,32)
            filter_ = 1
            x = np.expand_dims(self.data[i,:,:], axis=0)
            #x = np.expand_dims(x, axis=-1)
            #print(np.shape(x))
            intermediate_prediction_2=intermediate_model_2.predict(x)
            intermediate_prediction_2 = np.squeeze(intermediate_prediction_2[:,:,filter_], axis=0)
            
            
            
            fourier = np.fft.fft(intermediate_prediction_2)
            frequencies = np.fft.fftfreq(len(t), 0.004)  # where 0.005 is the inter-sample time difference
            positive_frequencies = frequencies[np.where(frequencies >= 0)] 
            #print(np.shape(frequencies)) 
            #print(np.shape(fourier))             
            #magnitudes = abs(fourier[np.where(frequencies >= 0)])  # magnitude spectrum
            #print(np.shape(magnitudes))  
            magnitudes = abs(fourier[0:int(np.shape(frequencies)[0]/2)])
            peak_frequency[i] = int(np.argmax(magnitudes)+1)
        #print(peak_frequency)
        #print(np.bincount(peak_frequency))
        np.savetxt('train_begin_2nd.txt', peak_frequency, delimiter=', ')
        
            
        #print(type(peak_frequency))
        #print(type(np.bincount(peak_frequency)))        
            
        #size_=range(0,np.shape(np.bincount(peak_frequency))[0])
        #print(np.shape(size_))    
        #print(np.bincount(peak_frequency))    
        #plt.bar(size_,np.bincount(peak_frequency),width=1)
        #plt.show()


        
        #x = np.expand_dims(self.data, axis=0)
        x = self.data
        x = np.expand_dims(self.data[0,:,:], axis=0)


        
        output_1 = self.model.get_layer('conv1d').output
        intermediate_model_1=tf.keras.models.Model(inputs=self.model.input,outputs=output_1)
        intermediate_prediction_1=intermediate_model_1.predict(x)
        np.savetxt('train_begin_0st_1sample.txt', intermediate_prediction_1[0,:,0], delimiter=', ')
        np.savetxt('train_begin_0st_2sample.txt', intermediate_prediction_1[0,:,10], delimiter=', ')
        np.savetxt('train_begin_0st_3sample.txt', intermediate_prediction_1[0,:,20], delimiter=', ')
        # #print(np.shape(intermediate_prediction))
        # plt.plot(intermediate_prediction_1[0,:,0])
        # plt.plot(intermediate_prediction_1[0,:,10])
        # plt.plot(intermediate_prediction_1[0,:,20])
        plt.show()
        output_2 = self.model.get_layer('sparseconnect_cnn_layer_1d').output
        intermediate_model_2=tf.keras.models.Model(inputs=self.model.input,outputs=output_2)
        intermediate_prediction_2=intermediate_model_2.predict(x)
        np.savetxt('train_begin_1st_1sample.txt', intermediate_prediction_2[0,:,0], delimiter=', ')
        np.savetxt('train_begin_1st_2sample.txt', intermediate_prediction_2[0,:,10], delimiter=', ')
        np.savetxt('train_begin_1st_3sample.txt', intermediate_prediction_2[0,:,20], delimiter=', ')
        # output_2 = self.model.get_layer('sparseconnect_cnn_layer_1d_1').output
        # intermediate_model_2=tf.keras.models.Model(inputs=self.model.input,outputs=output_2)
        # intermediate_prediction_2=intermediate_model_2.predict(x)
        # print(np.shape(intermediate_prediction_2))
        # plt.plot(intermediate_prediction_2[0,:,0])
        # plt.plot(intermediate_prediction_2[0,:,10])
        # plt.plot(intermediate_prediction_2[0,:,20])
        plt.show()
        output_3 = self.model.get_layer('sparseconnect_cnn_layer_1d_1').output
        intermediate_model_3=tf.keras.models.Model(inputs=self.model.input,outputs=output_3)
        intermediate_prediction_3=intermediate_model_3.predict(x)
        np.savetxt('train_begin_2nd_1sample.txt', intermediate_prediction_3[0,:,0], delimiter=', ')
        np.savetxt('train_begin_2nd_2sample.txt', intermediate_prediction_3[0,:,10], delimiter=', ')
        np.savetxt('train_begin_2nd_3sample.txt', intermediate_prediction_3[0,:,20], delimiter=', ')        
        # output_3 = self.model.get_layer('sparseconnect_cnn_layer_1d_2').output
        # intermediate_model_3=tf.keras.models.Model(inputs=self.model.input,outputs=output_3)
        # intermediate_prediction_3=intermediate_model_3.predict(x)
        # print(np.shape(intermediate_prediction_3))
        # plt.plot(intermediate_prediction_3[0,:,0])
        # plt.plot(intermediate_prediction_3[0,:,20])
        # plt.plot(intermediate_prediction_3[0,:,40])
        # plt.show()
        
    def on_train_end(self, epoch, logs=None):

        
        #x = np.expand_dims(self.data, axis=0)
        x = self.data
        print(np.shape(self.data))
        
        Fs=250.
        t = [i*1./Fs for i in range(200)]
        points=500
        
        
        peak_frequency = np.zeros([points], dtype=np.int64)
       
        # FFT of intermediate prediction
        for i in range(0,points):
            output_0 = self.model.get_layer('conv1d').output
            intermediate_model_0=tf.keras.models.Model(inputs=self.model.input,outputs=output_0)
            #print(np.shape(x[i,:,:]))
            #filter_ = np.random.randint(1,32)
            filter_ = 1
            x = np.expand_dims(self.data[i,:,:], axis=0)
            #x = np.expand_dims(x, axis=-1)
            #print(np.shape(x))
            intermediate_prediction_0=intermediate_model_0.predict(x)
            intermediate_prediction_0 = np.squeeze(intermediate_prediction_0[:,:,filter_], axis=0)
            fourier = np.fft.fft(intermediate_prediction_0)
            frequencies = np.fft.fftfreq(len(t), 0.004)  # where 0.005 is the inter-sample time difference
            #positive_frequencies = frequencies[np.where(frequencies >= 0)] 
            #print(np.shape(frequencies)) 
            #print(np.shape(fourier))             
            
            magnitudes = abs(fourier[0:int(np.shape(frequencies)[0]/2)])
            #magnitudes = abs(fourier[np.where(frequencies >= 0)])  # magnitude spectrum
            #print(np.shape(magnitudes))  
            peak_frequency[i] = int(np.argmax(magnitudes)+1)
        
        np.savetxt('train_end_0st.txt', peak_frequency, delimiter=', ')
        
        peak_frequency = np.zeros([points], dtype=np.int64)
        
        
        ## FFT of intermediate prediction
        for i in range(0,points):
            output_1 = self.model.get_layer('sparseconnect_cnn_layer_1d').output
            intermediate_model_1=tf.keras.models.Model(inputs=self.model.input,outputs=output_1)
            #print(np.shape(x[i,:,:]))
            #filter_ = np.random.randint(1,32)
            filter_ = 1
            x = np.expand_dims(self.data[i,:,:], axis=0)
            #x = np.expand_dims(x, axis=-1)
            #print(np.shape(x))
            intermediate_prediction_1=intermediate_model_1.predict(x)
            intermediate_prediction_1 = np.squeeze(intermediate_prediction_1[:,:,filter_], axis=0)
            fourier = np.fft.fft(intermediate_prediction_1)
            frequencies = np.fft.fftfreq(len(t), 0.004)  # where 0.005 is the inter-sample time difference
            #positive_frequencies = frequencies[np.where(frequencies >= 0)] 
            #print(np.shape(frequencies)) 
            #print(np.shape(fourier))             
            #magnitudes = abs(fourier[np.where(frequencies >= 0)])  # magnitude spectrum
            magnitudes = abs(fourier[0:int(np.shape(frequencies)[0]/2)])
            #print(np.shape(magnitudes))  
            peak_frequency[i] = int(np.argmax(magnitudes)+1)
        
        np.savetxt('train_end_1st.txt', peak_frequency, delimiter=', ')
        #print(peak_frequency)
        #print(np.bincount(peak_frequency))
        
            
        #print(type(peak_frequency))
        #print(type(np.bincount(peak_frequency)))        
            
        #size_=range(0,np.shape(np.bincount(peak_frequency))[0])
        #print(np.shape(size_))    
        #print(np.bincount(peak_frequency))    
        #plt.bar(size_,np.bincount(peak_frequency),width=1)
        #plt.show()
        
        ### second layer
        ## FFT of intermediate prediction
        Fs=125.
        peak_frequency = np.zeros([points], dtype=np.int64)
        for i in range(0,points):
            output_2 = self.model.get_layer('sparseconnect_cnn_layer_1d_1').output
            intermediate_model_2=tf.keras.models.Model(inputs=self.model.input,outputs=output_2)
            
            #print(np.shape(x[i,:,:]))
            #filter_ = np.random.randint(1,32)
            filter_ = 1
            x = np.expand_dims(self.data[i,:,:], axis=0)
            #x = np.expand_dims(x, axis=-1)
            #print(np.shape(x))
            intermediate_prediction_2=intermediate_model_2.predict(x)
            intermediate_prediction_2 = np.squeeze(intermediate_prediction_2[:,:,filter_], axis=0)
            
            
            
            fourier = np.fft.fft(intermediate_prediction_2)
            frequencies = np.fft.fftfreq(len(t), 0.004)  # where 0.005 is the inter-sample time difference
            #positive_frequencies = frequencies[np.where(frequencies >= 0)] 
            #print(np.shape(frequencies)) 
            #print(np.shape(fourier))             
            #magnitudes = abs(fourier[np.where(frequencies >= 0)])  # magnitude spectrum
            magnitudes = abs(fourier[0:int(np.shape(frequencies)[0]/2)])
            #print(np.shape(magnitudes))  
            peak_frequency[i] = int(np.argmax(magnitudes)+1)
        #print(peak_frequency)
        #print(np.bincount(peak_frequency))
        
        np.savetxt('train_end_2nd.txt', peak_frequency, delimiter=', ')
        
            
        #print(type(peak_frequency))
        #print(type(np.bincount(peak_frequency)))        
            
        #size_=range(0,np.shape(np.bincount(peak_frequency))[0])
        #print(np.shape(size_))    
        #print(np.bincount(peak_frequency))    
        #plt.bar(size_,np.bincount(peak_frequency),width=1)
        #plt.show()
        
        
        
        x = self.data
        x = np.expand_dims(self.data[0,:,:], axis=0)


         
        
        output_1 = self.model.get_layer('conv1d').output
        intermediate_model_1=tf.keras.models.Model(inputs=self.model.input,outputs=output_1)
        intermediate_prediction_1=intermediate_model_1.predict(x)
        np.savetxt('train_end_0st_1sample.txt', intermediate_prediction_1[0,:,0], delimiter=', ')
        np.savetxt('train_end_0st_2sample.txt', intermediate_prediction_1[0,:,10], delimiter=', ')
        np.savetxt('train_end_0st_3sample.txt', intermediate_prediction_1[0,:,20], delimiter=', ')
        # #print(np.shape(intermediate_prediction))
        # plt.plot(intermediate_prediction_1[0,:,0])   # filter 0
        # plt.plot(intermediate_prediction_1[0,:,10])   # filter 10
        # plt.plot(intermediate_prediction_1[0,:,20])   # filter 20
        # plt.show()
        
        output_1 = self.model.get_layer('sparseconnect_cnn_layer_1d').output
        intermediate_model_1=tf.keras.models.Model(inputs=self.model.input,outputs=output_1)
        intermediate_prediction_1=intermediate_model_1.predict(x)
        np.savetxt('train_end_1st_1sample.txt', intermediate_prediction_1[0,:,0], delimiter=', ')
        np.savetxt('train_end_1st_2sample.txt', intermediate_prediction_1[0,:,10], delimiter=', ')
        np.savetxt('train_end_1st_3sample.txt', intermediate_prediction_1[0,:,20], delimiter=', ')
        
        # output_2 = self.model.get_layer('sparseconnect_cnn_layer_1d_1').output
        # intermediate_model_2=tf.keras.models.Model(inputs=self.model.input,outputs=output_2)
        # intermediate_prediction_2=intermediate_model_2.predict(x)
        # print(np.shape(intermediate_prediction_2))
        # plt.plot(intermediate_prediction_2[0,:,0])
        # plt.plot(intermediate_prediction_2[0,:,10])
        # plt.plot(intermediate_prediction_2[0,:,20])
        # plt.show()
        
        output_1 = self.model.get_layer('sparseconnect_cnn_layer_1d_1').output
        intermediate_model_1=tf.keras.models.Model(inputs=self.model.input,outputs=output_1)
        intermediate_prediction_1=intermediate_model_1.predict(x)
        np.savetxt('train_end_2nd_1sample.txt',intermediate_prediction_1[0,:,0], delimiter=', ')
        np.savetxt('train_end_2nd_2sample.txt',intermediate_prediction_1[0,:,10], delimiter=', ')
        np.savetxt('train_end_2nd_3sample.txt',intermediate_prediction_1[0,:,20], delimiter=', ')
        
        # output_3 = self.model.get_layer('sparseconnect_cnn_layer_1d_2').output
        # intermediate_model_3=tf.keras.models.Model(inputs=self.model.input,outputs=output_3)
        # intermediate_prediction_3=intermediate_model_3.predict(x)
        # print(np.shape(intermediate_prediction_3))
        # plt.plot(intermediate_prediction_3[0,:,0])
        # plt.plot(intermediate_prediction_3[0,:,20])
        # plt.plot(intermediate_prediction_3[0,:,40])
        # plt.show()
        

        
        
   
        
