import scipy.io
import numpy as np
import fullduplex as fd
from keras.models import Model
from keras.layers import Layer, Input, Conv1D, LSTM, Flatten, Dense
from keras.optimizers import Adam
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# 禁用 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ======== 复值神经网络架构 (CVNN) ========
class ComplexReLU(Layer):
    """ 复数 ReLU (CReLU) """
    def call(self, inputs):
        return tf.complex(tf.nn.relu(tf.math.real(inputs)), tf.nn.relu(tf.math.imag(inputs)))

class ComplexConv1D(Layer):
    """ 复数卷积层 """
    def __init__(self, filters, kernel_size, activation=None, **kwargs):
        super(ComplexConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

    def build(self, input_shape):
        self.real_kernel = self.add_weight(shape=(self.kernel_size, input_shape[-1], self.filters),
                                           initializer="glorot_uniform", trainable=True, name="real_kernel")
        self.imag_kernel = self.add_weight(shape=(self.kernel_size, input_shape[-1], self.filters),
                                           initializer="glorot_uniform", trainable=True, name="imag_kernel")

    def call(self, inputs):
        real = tf.math.real(inputs)
        imag = tf.math.imag(inputs)

        real_output = tf.nn.conv1d(real, self.real_kernel, stride=1, padding='SAME') - tf.nn.conv1d(imag, self.imag_kernel, stride=1, padding='SAME')
        imag_output = tf.nn.conv1d(real, self.imag_kernel, stride=1, padding='SAME') + tf.nn.conv1d(imag, self.real_kernel, stride=1, padding='SAME')

        output = tf.complex(real_output, imag_output)
        return output

class ComplexLSTM(Layer):
    """ 复数 LSTM 层 """
    def __init__(self, units, **kwargs):
        super(ComplexLSTM, self).__init__(**kwargs)
        self.units = units
        self.real_lstm = LSTM(units, return_sequences=False)
        self.imag_lstm = LSTM(units, return_sequences=False)

    def call(self, inputs):
        real = tf.math.real(inputs)
        imag = tf.math.imag(inputs)

        real_output = self.real_lstm(real) - self.imag_lstm(imag)
        imag_output = self.real_lstm(imag) + self.imag_lstm(real)

        return tf.complex(real_output, imag_output)

class ComplexDense(Layer):
    """ 复数全连接层 """
    def __init__(self, units, activation=None, **kwargs):
        super(ComplexDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.real_kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                           initializer="glorot_uniform", trainable=True, name="real_kernel")
        self.imag_kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                           initializer="glorot_uniform", trainable=True, name="imag_kernel")

    def call(self, inputs):
        real = tf.math.real(inputs)
        imag = tf.math.imag(inputs)

        real_output = tf.matmul(real, self.real_kernel) - tf.matmul(imag, self.imag_kernel)
        imag_output = tf.matmul(real, self.imag_kernel) + tf.matmul(imag, self.real_kernel)

        output = tf.complex(real_output, imag_output)
        return output

# ======== 读取数据 ========
params = {
    'samplingFreqMHz': 20,
    'hSILen': 13,
    'pamaxordercanc': 7,
    'trainingRatio': 0.9,
    'dataOffset': 14,
    'nHidden': 20,
    'nEpochs': 60,
    'learningRate': 0.0005,
    'batchSize': 256,
    'convFilters': 32,
    'filterSize': 15,
    'lstmUnits': 32,
    'dropoutRate': 0.09
}
nHidden = params['nHidden']
nEpochs = params['nEpochs']
x, y, noise, measuredNoisePower = fd.loadData('data/fdTestbedData20MHz10dBm', params)
chanLen = params['hSILen']

# ======== 复值神经网络模型 ========
inputs = Input(shape=(chanLen, 1), dtype=tf.complex64)
conv = ComplexConv1D(filters=params['convFilters'], kernel_size=params['filterSize'])(inputs)
act = ComplexReLU()(conv)
lstm = ComplexLSTM(units=params['lstmUnits'])(act)
output = ComplexDense(units=1)(lstm)  # 输出复数值

model = Model(inputs, output)
adam = Adam(lr=params['learningRate'])
model.compile(loss="mse", optimizer=adam)

print("CVNN 参数数量:", model.count_params())

# ======== 数据预处理 ========
trainingSamples = int(np.floor(x.size * params['trainingRatio']))
x_train, y_train = x[:trainingSamples], y[:trainingSamples]
x_test, y_test = x[trainingSamples:], y[trainingSamples:]

# 线性自干扰消除

hLin = fd.SIestimationLinear(x_train, y_train, params)
yCanc = fd.SIcancellationLinear(x_train, hLin, params)

y_train = (y_train - yCanc)
yVar = np.var(y_train)
y_train = y_train/np.sqrt(yVar)

# 重新构造复数输入
x_train = np.reshape([x_train[i:i+chanLen] for i in range(x_train.size-chanLen)], (x_train.size-chanLen, chanLen, 1)).astype(np.complex64)
y_train = np.reshape(y_train[chanLen:], (y_train.size-chanLen, 1))

# 处理测试数据
yOrig = y_test
yCanc = fd.SIcancellationLinear(x_test, hLin, params)
y_test = (y_test - yCanc) / np.sqrt(yVar)

x_test = np.reshape([x_test[i:i+chanLen] for i in range(x_test.size-chanLen)], (x_test.size-chanLen, chanLen, 1)).astype(np.complex64)
y_test = np.reshape(y_test[chanLen:], (y_test.size-chanLen, 1))


# ======== 训练 ========
history = model.fit(x_train, y_train, epochs=params['nEpochs'], batch_size=params['batchSize'], validation_data=(x_test, y_test))

# ======== 预测和评估 ========
yCancNonLin = model.predict(x_test)
##### Evaluation #####
# Get correctly shaped test and cancellation data
y_test = np.squeeze(yOrig[chanLen:])
yCanc = np.squeeze(yCanc[chanLen:])
yCancNonLin = np.squeeze(yCancNonLin)


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

