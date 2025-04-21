import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# 設定隨機種子以確保結果可重現
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# 檢查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 建立結果目錄
os.makedirs('results', exist_ok=True)

# 讀取資料
print("讀取資料...")
df = pd.read_csv('training.csv')
print(f"資料維度: {df.shape}")

# 檢查數據集的列
print("\n顯示前10個列名:")
for i, col in enumerate(df.columns[:10]):
    print(f"{i}: {col}")

print("\n顯示隨機抽樣的10個列名:")
sample_cols = np.random.choice(df.columns, 10, replace=False)
for i, col in enumerate(sample_cols):
    print(f"{i}: {col}")

# 根據未來30天漲幅定義target
print("\n根據未來30天漲幅定義飆股標籤...")

# 嘗試尋找收盤價欄位
close_col = '季IFRS財報_季底收盤價(元)'
print(f"使用收盤價欄位: {close_col}")

# 計算未來30天最高收盤價
future_days = 30
df['future_max_price'] = df[close_col].shift(-1).rolling(window=future_days, min_periods=1).max()
df['price_change_pct'] = (df['future_max_price'] - df[close_col]) / df[close_col]

# 設定target：未來30天漲幅達10%以上為1，否則為0
threshold = 0.1
df['target'] = (df['price_change_pct'] >= threshold).astype(int)

print("飆股比例分布:")
print(df['target'].value_counts(normalize=True))

# 為了處理嚴重不平衡，我們可以嘗試下採樣多數類
# 首先，將數據分為多數類和少數類
majority_class = df[df['target'] == 1]
minority_class = df[df['target'] == 0]

print(f"多數類數量: {len(majority_class)}")
print(f"少數類數量: {len(minority_class)}")

# 下採樣多數類，保留所有少數類
# 假設我們希望多數類樣本數是少數類的3倍
sampling_ratio = 3
sampled_majority = majority_class.sample(n=min(len(majority_class), len(minority_class) * sampling_ratio), random_state=42)
df_balanced = pd.concat([sampled_majority, minority_class])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # 打亂順序

print("下採樣後的數據分布:")
print(df_balanced['target'].value_counts(normalize=True))
print(f"下採樣後的數據維度: {df_balanced.shape}")

# 特徵選擇：選擇數值型特徵
print("\n選擇數值型特徵...")
# 排除非數值型特徵和目標變量
exclude_cols = ['target', 'future_max_price', 'price_change_pct']
if 'date' in df_balanced.columns:
    exclude_cols.append('date')

features = [col for col in df_balanced.columns 
            if col not in exclude_cols 
            and np.issubdtype(df_balanced[col].dtype, np.number)]

# 選擇前100個特徵 (可以根據需要調整)
if len(features) > 100:
    print(f"特徵太多 ({len(features)}), 只選擇前100個")
    features = features[:100]

print(f"使用的特徵數量: {len(features)}")
print(f"特徵示例: {features[:5]}")

# 處理缺失值
print("\n處理缺失值...")
df_balanced[features] = df_balanced[features].fillna(df_balanced[features].median())

# 使用RobustScaler進行特徵縮放
print("\n標準化特徵...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_balanced[features])

# 創建時間序列數據
print("\n建立時間序列資料...")
sequence_length = 10  # 調整為較短的序列長度，適合稀疏數據

def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, df_balanced['target'].values, sequence_length)
print(f"序列資料維度: {X_seq.shape}")
print(f"標籤維度: {y_seq.shape}")

# 切分訓練集和測試集
train_ratio = 0.8
train_size = int(len(X_seq) * train_ratio)

X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

print(f"訓練集大小: {X_train.shape}")
print(f"測試集大小: {X_test.shape}")

# 定義數據集和數據載入器
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 64
train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定義簡化的LSTM模型
class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(SimpleLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最後一個時間步
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

# 初始化模型
model = SimpleLSTMModel(input_size=len(features)).to(device)
print(model)

# 定義損失函數和優化器
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
print("\n開始訓練模型...")
num_epochs = 20  # 減少訓練輪數，先看結果

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")
    
    # 每5個epoch評估一次
    if (epoch + 1) % 5 == 0:
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                y_pred.extend(outputs.cpu().numpy() >= 0.5)
                y_true.extend(y_batch.numpy())
        
        f1 = f1_score(y_true, y_pred)
        print(f"Validation F1 Score: {f1:.4f}")

# 保存模型
torch.save(model.state_dict(), 'results/stock_lstm_model.pth')

# 最終評估
model.eval()
y_true, y_pred_prob = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        y_pred_prob.extend(outputs.cpu().numpy())
        y_true.extend(y_batch.numpy())

y_true = np.array(y_true).flatten()
y_pred_prob = np.array(y_pred_prob).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

# 計算評估指標
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n最終評估結果:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 繪製混淆矩陣
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('results/confusion_matrix.png')

print("\n訓練完成！模型已保存到 results/stock_lstm_model.pth")