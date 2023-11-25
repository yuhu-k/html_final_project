import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import torch.nn as nn



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

data = pd.read_csv('stage1_input.csv')
# 載入模型
model = Dual_LSTM(data.shape[1],hidden_size1=128,hidden_size2=64,num_layers=2,output_size=1).to('cuda')  # 這裡替換成你的模型類別
model.load_state_dict(torch.load('model.pth'))
model.eval()  # 設定模型為評估模式

data = pd.read_csv('input.csv')

tensor_data = ToTensor()(data)

# 將資料轉換為模型期望的 device (CPU 或 GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor_data = tensor_data.to(device)

# 進行預測
with torch.no_grad():
    output = model(tensor_data)

# 將 PyTorch 張量轉換為 NumPy 陣列
output_np = output.cpu().numpy()

# 將 NumPy 陣列轉換為 Pandas DataFrame
output_df = pd.DataFrame(output_np)  # 替換成你的列名

# 將 DataFrame 寫入 CSV 檔案
output_df.to_csv('drive/MyDrive/html_final_project/output.csv', index=False)