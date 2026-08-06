import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os

class TEGDigitalTwinNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, output_dim=2):
        super(TEGDigitalTwinNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid() # Saídas entre 0.0 e 1.0
        )
        
    def forward(self, x):
        return self.network(x)

def train_model():
    dataset_path = "teg_dataset.csv"
    if not os.path.exists(dataset_path):
        from generate_dataset import generate_teg_dataset
        generate_teg_dataset()

    df = pd.read_csv(dataset_path)
    
    X = df[["t_quente", "t_frio", "v_real", "v_ideal", "p_real_mW", "p_ideal_mW"]].values
    y = df[["score_degradacao", "eficiencia_relativa"]].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = TEGDigitalTwinNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    epochs = 40
    print("🚀 Treinando modelo PyTorch...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Época [{epoch+1}/{epochs}] - Perda (MSE): {epoch_loss/len(dataloader):.6f}")

    os.makedirs("models", exist_ok=True)

    # Salva o arquivo dentro da pasta mapeada
    torch.save(model.state_dict(), "models/teg_model.pt")
    print("💾 Modelo salvo com sucesso em 'models/teg_model.pt'!")

if __name__ == "__main__":
    train_model()