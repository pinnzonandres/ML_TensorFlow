# -*- coding: utf-8 -*-
"""MultipleLinearRegressor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gdikNT52GD32_t2EnemgR4sHkjK5kFh0
"""

#Instalar Paquetes
import tensorflow as tf


import numpy as np
import pandas as pd
import datetime
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

#Descargamos los datos

df= pdr.get_data_yahoo('AAPL', start= datetime.datetime(2019,1,1), end = datetime.datetime(2019,12,31))
df.head()

df.describe

fig = plt.figure()
ax=Axes3D(fig)
#ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['High'],df['Open'],df['Adj Close'],c='blue',marker='o',alpha=0.5)
ax.set_xlabel('High')
ax.set_ylabel('Open')
ax.set_zlabel('AdjClose')
plt.show()

#Normalizar los datos
mean = df.mean()
std =df.std()
df_norm= (df-mean) / std
df_norm.head()

df_norm.describe()

#Segmentamos los datos para las variables que queremos usar
feature_names=['High','Low','Open']
data_x= df_norm[feature_names]
data_y= df_norm['Adj Close']
print ('input_shape X_Data', data_x.shape)
print ('input_shape Y_Data', data_y.shape)

#Separamos los datos
train_x , test_x ,train_Y, test_Y = train_test_split( data_x , data_y , test_size=0.2 )

#Preparamos los datos a tensorflow
train_x = tf.constant( train_x , dtype=tf.float32 )
train_Y = tf.constant( train_Y , dtype=tf.float32 )

test_X = tf.constant( test_x , dtype=tf.float32 ) 
test_Y = tf.constant( test_Y , dtype=tf.float32 )

#Definimos nuestras funciones y no sessiones para tf=2.00
def mean_squared_error( Y , y_pred ):
    return tf.reduce_mean( tf.square( y_pred - Y ) )

def mean_squared_error_deriv( Y , y_pred ):
    return tf.reshape( tf.reduce_mean( 2 * ( y_pred - Y ) ) , [ 1 , 1 ] )
    
def h ( X , weights , bias ):
    return tf.tensordot( X , weights , axes=1 ) + bias

#Parametros del entrenamiento
num_epochs = 10
num_samples = train_x.shape[0]
batch_size = 10
learning_rate = 0.01

#Preparamos los datos como un iterator para entrenarlo
dataset = tf.data.Dataset.from_tensor_slices(( train_x , train_Y )) 
dataset = dataset.shuffle( 500 ).repeat( num_epochs ).batch( batch_size )
iterator = dataset.__iter__()

num_features = train_x.shape[1]
weights = tf.random.normal( ( num_features , 1 ) ) 
bias = 0

epochs_plot = list()
loss_plot = list()

for i in range( num_epochs ) :
    
    epoch_loss = list()
    for b in range( int(num_samples/batch_size) ):
        x_batch , y_batch = iterator.get_next()
   
        output = h( x_batch , weights , bias ) 
        loss = epoch_loss.append( mean_squared_error( y_batch , output ).numpy() )
    
        dJ_dH = mean_squared_error_deriv( y_batch , output)
        dH_dW = x_batch
        dJ_dW = tf.reduce_mean( dJ_dH * dH_dW )
        dJ_dB = tf.reduce_mean( dJ_dH )
    
        weights -= ( learning_rate * dJ_dW )
        bias -= ( learning_rate * dJ_dB ) 
        
    loss = np.array( epoch_loss ).mean()
    epochs_plot.append( i + 1 )
    loss_plot.append( loss ) 
    
    print( 'Loss is {}'.format( loss ) )

import matplotlib.pyplot as plt
plt.plot( epochs_plot , loss_plot ) 
plt.show()

output = h( test_X , weights , bias ) 
labels = test_Y

accuracy_op = tf.metrics.MeanAbsoluteError() 
accuracy_op.update_state( labels , output )
print( 'Mean Absolute Error = {}'.format( accuracy_op.result().numpy() ) )