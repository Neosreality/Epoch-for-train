import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 1. Örnek Veri
data = np.random.rand(100, 10).astype(np.float32)  # 100 örnek, 10 özellik
labels = np.random.randint(0, 2, 100).astype(np.float32)  # 0 ve 1 sınıfları için rastgele etiketler

train_data = torch.tensor(data)
train_labels = torch.tensor(labels)

dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# 2. Model Tanımı
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Model oluştur
model = SimpleNN(input_size=10, hidden_size=5, output_size=1)  # 10 giriş, 5 gizli, 1 çıktı

# 3. Kayıp Fonksiyonu ve Optimizatör
criterion = nn.BCEWithLogitsLoss()  # İkili sınıflandırma
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Eğitim Döngüsü
num_epochs = 20
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

