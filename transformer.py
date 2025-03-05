import scipy.io
import numpy as np
import fullduplex as fd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

# 禁用 GPU（如数据集较小）
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# **系统参数**
params = {
    'samplingFreqMHz': 20,
    'hSILen': 13,  # 自干扰信道长度
    'pamaxordercanc': 7,  # 最大PA非线性阶数
    'trainingRatio': 0.9,  # 训练数据比例
    'dataOffset': 14,  # 发射机接收机失配补偿
    'nEpochs': 40,  # 训练轮数
    'learningRate': 0.0005,  # 学习率
    'batchSize': 32,  # 批量大小
    'd_model': 64,  # Transformer嵌入维度
    'num_heads': 4,  # 多头注意力头数
    'num_layers': 3,  # Transformer编码层数
    'dim_feedforward': 128  # 前馈网络维度
}

##### **加载数据** #####
x, y, noise, measuredNoisePower = fd.loadData('data/fdTestbedData20MHz10dBm', params)

# 获取信道长度
chanLen = params['hSILen']

# 划分训练和测试数据
trainingSamples = int(np.floor(x.size * params['trainingRatio']))
x_train, y_train = x[:trainingSamples], y[:trainingSamples]
x_test, y_test = x[trainingSamples:], y[trainingSamples:]

# 进行线性自干扰消除
hLin = fd.SIestimationLinear(x_train, y_train, params)
yCanc = fd.SIcancellationLinear(x_train, hLin, params)

# 只保留非线性干扰部分
y_train = y_train - yCanc
yVar = np.var(y_train)
y_train = y_train / np.sqrt(yVar)  # 归一化处理

yCanc = fd.SIcancellationLinear(x_test, hLin, params) #线性自干扰分量
yOrig = y_test
y_test = y_test - yCanc
y_test = y_test/np.sqrt(yVar)


# **拆分输入数据（实部+虚部）**
def prepare_data(x, y, chanLen):
    x_real = np.array([x[i:i + chanLen].real for i in range(x.size - chanLen)])
    x_imag = np.array([x[i:i + chanLen].imag for i in range(x.size - chanLen)])
    x_input = np.concatenate((x_real, x_imag), axis=1)  # 拼接实部和虚部

    y_real = np.array(y[chanLen:].real)
    y_imag = np.array(y[chanLen:].imag)

    return torch.tensor(x_input, dtype=torch.float32), torch.tensor(y_real, dtype=torch.float32), torch.tensor(y_imag,
                                                                                                               dtype=torch.float32)


x_train, y_train_real, y_train_imag = prepare_data(x_train, y_train, chanLen)
x_test, y_test_real, y_test_imag = prepare_data(x_test, y_test, chanLen)


# 转换为 DataLoader 格式
train_dataset = TensorDataset(x_train, y_train_real, y_train_imag)
train_loader = DataLoader(train_dataset, batch_size=params['batchSize'], shuffle=True)

test_dataset = TensorDataset(x_test, y_test_real, y_test_imag)
test_loader = DataLoader(test_dataset, batch_size=params['batchSize'], shuffle=False)


##### **定义 Transformer 模型** #####
class SICTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dim_feedforward):
        super(SICTransformer, self).__init__()
        self.input_layer = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        self.output_layer_real = nn.Linear(d_model, 1)
        self.output_layer_imag = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = x.unsqueeze(1)  # Transformer 需要 batch_size, seq_len, feature_dim 结构
        encoded = self.transformer(x)
        encoded = encoded.squeeze(1)
        output_real = self.output_layer_real(encoded)
        output_imag = self.output_layer_imag(encoded)
        return output_real, output_imag


# **初始化模型**
model = SICTransformer(input_dim=2 * chanLen, d_model=params['d_model'], num_heads=params['num_heads'],
                       num_layers=params['num_layers'], dim_feedforward=params['dim_feedforward'])
optimizer = optim.Adam(model.parameters(), lr=params['learningRate'], weight_decay=1e-5)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

##### **训练模型** #####
train_losses = []
val_losses = []
def train_model(model, train_loader, test_loader, epochs):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learningRate'])

    for epoch in range(epochs):
        total_train_loss = 0.0
        total_val_loss = 0.0

        # **训练阶段**
        model.train()
        for x_batch, y_real_batch, y_imag_batch in train_loader:
            optimizer.zero_grad()
            output_real, output_imag = model(x_batch)
            loss_real = criterion(output_real.squeeze(), y_real_batch)
            loss_imag = criterion(output_imag.squeeze(), y_imag_batch)
            loss = loss_real + loss_imag
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # **验证阶段**
        model.eval()
        with torch.no_grad():
            for x_batch, y_real_batch, y_imag_batch in test_loader:
                output_real, output_imag = model(x_batch)
                loss_real = criterion(output_real.squeeze(), y_real_batch)
                loss_imag = criterion(output_imag.squeeze(), y_imag_batch)
                loss = loss_real + loss_imag
                total_val_loss += loss.item()

        # 计算平均损失
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")



train_model(model, train_loader, test_loader,epochs=params['nEpochs'])


##### **测试与性能分析** #####
def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        all_preds_real, all_preds_imag = [], []
        all_targets_real, all_targets_imag = [], []

        for x_batch, y_real_batch, y_imag_batch in test_loader:
            output_real, output_imag = model(x_batch)
            all_preds_real.extend(output_real.squeeze().tolist())
            all_preds_imag.extend(output_imag.squeeze().tolist())
            all_targets_real.extend(y_real_batch.tolist())
            all_targets_imag.extend(y_imag_batch.tolist())

    return np.array(all_preds_real), np.array(all_preds_imag), np.array(all_targets_real), np.array(all_targets_imag)



# **获取测试结果**
y_pred_real, y_pred_imag, y_test_real, y_test_imag = test_model(model, test_loader)

# **恢复完整信号**
y_pred_complex = y_pred_real + 1j * y_pred_imag
y_test_complex = y_test_real + 1j * y_test_imag

##### **计算信号功率 & 绘制 PSD** #####
y_test = yOrig[chanLen:]
yCanc = yCanc[chanLen:]
yCancNonLin = y_pred_complex  # Transformer 预测的非线性部分

# 归一化
scalingConst = np.power(10, -(measuredNoisePower - 10 * np.log10(np.mean(np.abs(noise) ** 2))) / 10)
noise /= np.sqrt(scalingConst)
y_test /= np.sqrt(scalingConst)
yCanc /= np.sqrt(scalingConst)
yCancNonLin /= np.sqrt(scalingConst)

# 绘制 PSD 并计算信号功率
noisePower, yTestPower, yTestLinCancPower, yTestNonLinCancPower = fd.plotPSD(y_test, yCanc, yCancNonLin, noise, params,
                                                                             'Transformer', yVar)

# **打印 SIC 性能**
print(f"Linear SI cancellation: {yTestPower - yTestLinCancPower:.2f} dB")
print(f"Non-linear SI cancellation: {yTestLinCancPower - yTestNonLinCancPower:.2f} dB")
print(f"Noise floor: {noisePower:.2f} dBm")
print(f"Distance from noise floor: {yTestNonLinCancPower - noisePower:.2f} dB")

plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(train_losses) + 1), -10 * np.log10(train_losses), 'bo-', label="Training Loss")
plt.plot(np.arange(1, len(val_losses) + 1), -10 * np.log10(val_losses), 'ro-', label="Validation Loss")
plt.ylabel("Self-Interference Cancellation (dB)")
plt.xlabel("Training Epoch")
plt.legend(loc="lower right")
plt.grid(which="major", alpha=0.25)
plt.xlim([0, params['nEpochs'] + 1])
plt.xticks(range(1, params['nEpochs'], 2))
plt.savefig("figures/TransformerSIC_Loss.pdf", bbox_inches="tight")
plt.show()