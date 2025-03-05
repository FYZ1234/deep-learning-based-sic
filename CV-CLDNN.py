import scipy.io
from scipy.signal import savgol_filter
import numpy as np
import fullduplex as fd
from keras.models import Model
from keras.layers import Dense, Input, Conv1D, LSTM, Flatten, Concatenate
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# This line disables the use of the GPU for training.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Define system parameters
params = {
    'samplingFreqMHz': 20,  # Sampling frequency
    'hSILen': 13,           # Self-interference channel length (M+L=13)
    'pamaxordercanc': 7,    # Maximum PA non-linearity order
    'trainingRatio': 0.9,   # Ratio of total samples to use for training
    'dataOffset': 14,       # Data offset
    'nHidden': 17,          # Number of hidden units in fully connected layer
    'nEpochs': 60,          # Number of training epochs
    'learningRate': 0.004,  # Learning rate
    'batchSize': 256,       # Batch size
    'convFilters': 25,      # Number of convolution filters
    'filterSize': 8,        # Filter size
    'lstmUnits': 32,        # Number of LSTM units
    'dropoutRate': 0.09     # Dropout rate for LSTM layer
}

##### Load and prepare data #####

x, y, noise, measuredNoisePower = fd.loadData('data/fdTestbedData20MHz10dBm', params)

# Get self-interference channel length
chanLen = params['hSILen']

# CV-CLDNN model creation
input_real = Input(shape=(chanLen, 1))
input_imag = Input(shape=(chanLen, 1))

# Convolutional layers
conv_real = Conv1D(filters=params['convFilters'], kernel_size=params['filterSize'], activation='relu')(input_real)
conv_imag = Conv1D(filters=params['convFilters'], kernel_size=params['filterSize'], activation='relu')(input_imag)

# Concatenate real and imaginary parts
concat = Concatenate()([conv_real, conv_imag])

# LSTM layer
lstm = LSTM(params['lstmUnits'], dropout=params['dropoutRate'], return_sequences=False)(concat)

# Fully connected layers for real and imaginary parts
output_real = Dense(1, activation='linear')(lstm)
output_imag = Dense(1, activation='linear')(lstm)

# Model definition
model = Model(inputs=[input_real, input_imag], outputs=[output_real, output_imag])
adam = Adam(learning_rate=params['learningRate'])
model.compile(loss="mse", optimizer=adam)

print("Total number of parameters for CV-CLDNN: {:d}".format(model.count_params()))

# Split into training and test sets
trainingSamples = int(np.floor(x.size * params['trainingRatio']))
x_train = x[:trainingSamples]
y_train = y[:trainingSamples]
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

# Prepare training and test data
x_train_real = np.reshape(np.array([x_train[i:i+chanLen].real for i in range(x_train.size-chanLen)]), (x_train.size-chanLen, chanLen, 1))
x_train_imag = np.reshape(np.array([x_train[i:i+chanLen].imag for i in range(x_train.size-chanLen)]), (x_train.size-chanLen, chanLen, 1))
y_train = np.reshape(y_train[chanLen:], (y_train.size-chanLen, 1))

# Prepare test data for NN
yCanc = fd.SIcancellationLinear(x_test, hLin, params)
yOrig = y_test
y_test = y_test - yCanc
y_test = y_test/np.sqrt(yVar)

x_test_real = np.reshape(np.array([x_test[i:i+chanLen].real for i in range(x_test.size-chanLen)]), (x_test.size-chanLen, chanLen, 1))
x_test_imag = np.reshape(np.array([x_test[i:i+chanLen].imag for i in range(x_test.size-chanLen)]), (x_test.size-chanLen, chanLen, 1))
y_test = np.reshape(y_test[chanLen:], (y_test.size-chanLen, 1))

##### Training #####
history = model.fit(
    [x_train_real, x_train_imag], [y_train.real, y_train.imag],
    epochs=params['nEpochs'],
    batch_size=params['batchSize'],
    verbose=2,
    validation_data=([x_test_real, x_test_imag], [y_test.real, y_test.imag])
)

##### Test #####
pred = model.predict([x_test_real, x_test_imag])
yCancNonLin = np.squeeze(pred[0] + 1j * pred[1], axis=1)

##### Evaluation #####
y_test = yOrig[chanLen:]
yCanc = yCanc[chanLen:]

noisePower = 10*np.log10(np.mean(np.abs(noise)**2))
scalingConst = np.power(10,-(measuredNoisePower-noisePower)/10)
noise /= np.sqrt(scalingConst)
y_test /= np.sqrt(scalingConst)
yCanc /= np.sqrt(scalingConst)
yCancNonLin /= np.sqrt(scalingConst)

noisePower, yTestPower, yTestLinCancPower, yTestNonLinCancPower = fd.plotPSD(
    y_test, yCanc, yCancNonLin, noise, params, 'NN', yVar
)

print('\n')
print('The linear SI cancellation is: {:.2f} dB'.format(yTestPower - yTestLinCancPower))
print('The non-linear SI cancellation is: {:.2f} dB'.format(yTestLinCancPower - yTestNonLinCancPower))
print('The noise floor is: {:.2f} dBm'.format(noisePower))
print('The distance from noise floor is: {:.2f} dB'.format(yTestNonLinCancPower - noisePower))

##### Plot Learning Curve #####
plt.plot(np.arange(1, len(history.history['loss']) + 1), -10 * np.log10(history.history['loss']), 'bo-')
plt.plot(np.arange(1, len(history.history['loss']) + 1), -10 * np.log10(history.history['val_loss']), 'ro-')
plt.ylabel('Self-Interference Cancellation (dB)')
plt.xlabel('Training Epoch')
plt.legend(['Training Frame', 'Test Frame'], loc='lower right')
plt.grid(which='major', alpha=0.25)
plt.xlim([0, params['nEpochs'] + 1])
plt.xticks(range(1, params['nEpochs'], 5))
plt.savefig('figures/CV-CLDNNconv.pdf', bbox_inches='tight')
plt.show()
