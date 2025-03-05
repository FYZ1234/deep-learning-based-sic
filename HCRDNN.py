import scipy.io
from scipy.signal import savgol_filter
import numpy as np
import fullduplex as fd
from keras.models import Model, Sequential
from keras.layers import Dense, Input, SimpleRNN, Dropout,Conv1D,Reshape,LSTM
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import math


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU:", gpus)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, using CPU instead.")


# This line disables the use of the GPU for training. The dataset is not large enough to get
# significant gains from GPU training and, in fact, sometimes training can even be slower on
# the GPU than on the CPU. Comment out to enable GPU use.

# Define system parameters
params = {
		'samplingFreqMHz': 20,	# Sampling frequency, required for correct scaling of PSD
		'hSILen': 13,			# Self-interference channel length
		'pamaxordercanc': 7,	# Maximum PA non-linearity order
		'trainingRatio': 0.9,	# Ratio of total samples to use for training
		'dataOffset': 14,		# Data offset to take transmitter-receiver misalignment into account
		'nHidden': 17,			# Number of hidden layers in NN
		'nEpochs': 80,			# Number of training epochs for NN training
		'learningRate': 0.0005,	# Learning rate for NN training
		'batchSize': 256,		# Batch size for NN training
		}

##### Load and prepare data #####

x, y, noise, measuredNoisePower = fd.loadData('data/fdTestbedData20MHz10dBm', params)

# Get self-interference channel length
chanLen = params['hSILen']

# Create feedforward NN using Keras
nHidden = params['nHidden']
nEpochs = params['nEpochs']
input = Input(shape=(2*chanLen,1))
print("Input layer shape:", input.shape)
conv1 = Conv1D(filters=2, kernel_size=12, strides=1, activation='relu')(input)
print("Conv1D layer output shape:", conv1.shape)
reshape_layer = Reshape((conv1.shape[1]*conv1.shape[2],1))(conv1)
print("reshape_layer output shape:", reshape_layer.shape)
rnn1 = SimpleRNN(64, return_sequences=True, activation='relu',dropout=0.1)(reshape_layer)
rnn2 = SimpleRNN(64, return_sequences=False, activation='relu',dropout=0.1)(rnn1)
print("SimpleRNN layer output shape:", rnn2.shape)
dense = Dense(64, activation='relu')(rnn2)
dropout1 = Dropout(0.2)(dense)
print("Dense layer output shape:", dense.shape)
output1 = Dense(1, activation='linear')(dense)
output2 = Dense(1, activation='linear')(dense)
model = Model(inputs=input, outputs=[output1, output2])
adam = Adam(learning_rate =params['learningRate'])
model.compile(loss = "mse", optimizer = adam)
print("Total number of real parameters to estimate for neural network based canceller: {:d}".format((2*chanLen+1)*nHidden + 2*(nHidden+1)+2*chanLen))

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.5,
                                                 patience=5,
                                                 min_lr=1e-6)

# Split into training and test sets
trainingSamples = int(np.floor(x.size*params['trainingRatio']))
x_train = x[0:trainingSamples]
y_train = y[0:trainingSamples]
x_test = x[trainingSamples:]
y_test = y[trainingSamples:]

##### Training #####
# Step 1: Estimate linear cancellation parameters and perform linear cancellation
hLin = fd.SIestimationLinear(x_train, y_train, params)
yCanc = fd.SIcancellationLinear(x_train, hLin, params)

# Normalize data for NN
yOrig = y_train
y_train = y_train - yCanc
yVar = np.var(y_train)
y_train = y_train/np.sqrt(yVar)

# Prepare training data for NN
x_train_real = np.reshape(np.array([x_train[i:i+chanLen].real for i in range(x_train.size-chanLen)]), (x_train.size-chanLen, chanLen))
x_train_imag = np.reshape(np.array([x_train[i:i+chanLen].imag for i in range(x_train.size-chanLen)]), (x_train.size-chanLen, chanLen))
x_train = np.zeros((x_train.size-chanLen, 2*chanLen))
x_train[:,0:chanLen] = x_train_real
x_train[:,chanLen:2*chanLen] = x_train_imag

y_train = np.reshape(y_train[chanLen:], (y_train.size-chanLen, 1))
print(x_train.shape)

# Prepare test data for NN
yCanc = fd.SIcancellationLinear(x_test, hLin, params)
yOrig = y_test
y_test = y_test - yCanc
y_test = y_test/np.sqrt(yVar)

x_test_real = np.reshape(np.array([x_test[i:i+chanLen].real for i in range(x_test.size-chanLen)]), (x_test.size-chanLen, chanLen))
x_test_imag = np.reshape(np.array([x_test[i:i+chanLen].imag for i in range(x_test.size-chanLen)]), (x_test.size-chanLen, chanLen))
x_test = np.zeros((x_test.size-chanLen, 2*chanLen))
x_test[:,0:chanLen] = x_test_real
x_test[:,chanLen:2*chanLen] = x_test_imag
x_test = np.expand_dims(x_test, axis=-1)
y_test = np.reshape(y_test[chanLen:], (y_test.size-chanLen, 1))

##### Training #####
# Step 2: train NN to do non-linear cancellation
history = model.fit(x_train, [y_train.real, y_train.imag], epochs = nEpochs, batch_size = params['batchSize'], verbose=2, validation_data= (x_test, [y_test.real, y_test.imag]),callbacks=[reduce_lr])

##### Test #####
# Do inference step
pred = model.predict(x_test)
yCancNonLin = np.squeeze(pred[0] + 1j*pred[1], axis=1)

##### Evaluation #####
# Get correctly shaped test and cancellation data
y_test = yOrig[chanLen:]
yCanc = yCanc[chanLen:]

# Calculate various signal powers
noisePower = 10*np.log10(np.mean(np.abs(noise)**2))
scalingConst = np.power(10,-(measuredNoisePower-noisePower)/10)
noise /= np.sqrt(scalingConst)
y_test /= np.sqrt(scalingConst)
yCanc /= np.sqrt(scalingConst)
yCancNonLin /= np.sqrt(scalingConst)

# Plot PSD and get signal powers
noisePower, yTestPower, yTestLinCancPower, yTestNonLinCancPower = fd.plotPSD(y_test, yCanc, yCancNonLin, noise, params, 'NN', yVar)

# Print cancellation performance
print('')
print('The linear SI cancellation is: {:.2f} dB'.format(yTestPower-yTestLinCancPower))
print('The non-linear SI cancellation is: {:.2f} dB'.format(yTestLinCancPower-yTestNonLinCancPower))
print('The noise floor is: {:.2f} dBm'.format(noisePower))
print('The distance from noise floor is: {:.2f} dB'.format(yTestNonLinCancPower-noisePower))

# Plot learning curve
plt.plot(np.arange(1,len(history.history['loss'])+1), -10*np.log10(history.history['loss']), 'bo-')
plt.plot(np.arange(1,len(history.history['loss'])+1), -10*np.log10(history.history['val_loss']), 'ro-')
plt.ylabel('Self-Interference Cancellation (dB)')
plt.xlabel('Training Epoch')
plt.legend(['Training Frame', 'Test Frame'], loc='lower right')
plt.grid(which='major', alpha=0.25)
plt.xlim([ 0, nEpochs+1 ])
plt.xticks(range(1,nEpochs,2))
plt.savefig('figures/NNconv.pdf', bbox_inches='tight')
plt.show()
