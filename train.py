import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn.functional as F

# 步骤 1: 导入必要的库

# 步骤 2: 加载和预处理数据

# 读取CSV文件
data_reader = pd.read_csv('data.csv', header=1, chunksize=100000)
data = pd.concat(data_reader, ignore_index=True)

# 划分特征和标签
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)



# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch的张量并将其移到CUDA设备上
X_train_tensor = torch.FloatTensor(X_train).to('cuda')
y_train_tensor = torch.FloatTensor(y_train).to('cuda')
X_test_tensor = torch.FloatTensor(X_test).to('cuda')
y_test_tensor = torch.FloatTensor(y_test).to('cuda')

# 将数据转换为PyTorch的Dataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 定义批处理大小
batch_size = 1024

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 步骤 3: 定义模型

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        output = self.linear(x)
        upbound = torch.max(x[:,-1].view(-1,1))
        output = torch.clamp(output,0,upbound)
        return output

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        out,_ = self.lstm(x)
        out = self.fc(out[:,:])

        upbound = torch.max(x[:,-1].view(-1,1))
        out = torch.clamp(out,0,upbound)
        return out

class Dual_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, output_size):
        super(Dual_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size1, hidden_size1)

        self.activation = nn.ReLU()

        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):

        out,_ = self.lstm1(x)
        out = self.fc1(out[:,:])

        out = self.activation(out)

        out,_ = self.lstm2(out)
        out = self.fc2(out[:,:])

        upbound = torch.max(x[:,-1].view(-1,1))
        out = torch.clamp(out,0,upbound)
        return out

# 步骤 4: 定义损失函数和优化器

# 创建模型并将其移到CUDA设备上
model = Dual_LSTM(X_train.shape[1],hidden_size1=128,hidden_size2=64,num_layers=2,output_size=1).to('cuda')

class CustomLoss(nn.Module):
  def __init__(self):
    super(CustomLoss, self).__init__()

  def forward(self, predictions, targets, inputs):
    input_col_3 = inputs[:,-1].view(-1,1)
    loss = 3 * (torch.abs(predictions - targets) / input_col_3) * \
     (torch.abs(targets/input_col_3 - 1/3) + torch.abs(targets/input_col_3 - 2/3))


    return loss.mean()
# 定义损失函数和优化器
criterion = CustomLoss()
criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 步骤 5: 训练模型

# 定义训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels, inputs)
            loss.backward()
            optimizer.step()

        # 打印每个epoch的损失
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

        # 保存模型和优化器状态
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, f'results/model_epoch_{epoch}.pth')

# 训练模型
train_model(model, train_loader, criterion, optimizer, num_epochs=5)

# 在测试集上评估模型
model.eval().cuda()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor, X_test_tensor)
    print(f'Test Loss: {test_loss.item()}')