import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projections
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        # Split heads
        q = q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.depth ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)

        # Concatenate heads and apply final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(output)
        return output, attention_weights

class OlympicModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(OlympicModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.mha = MultiHeadAttention(d_model=64, num_heads=8)
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = x.unsqueeze(1)
        x, attention_weights = self.mha(x, x, x)
        x = x.squeeze(1)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x, attention_weights

def train_and_evaluate_dl(X_train, X_test, y_train, y_test, epochs=100, batch_size=32, patience=10):
    """
    Trains and evaluates a deep learning model with early stopping.

    Args:
        X_train, X_test, y_train, y_test: Training and testing data.
        epochs: Maximum number of training epochs.
        batch_size: Batch size for training.
        patience: Number of epochs to wait for improvement before early stopping.

    Returns:
        dl_model: The trained deep learning model.
    """

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    input_dim = X_train.shape[1]
    dl_model = OlympicModel(input_dim)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(dl_model.parameters(), lr=0.001, weight_decay=1e-5) # L2 regularization
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    train_losses = []
    for epoch in range(epochs):
        dl_model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs, _ = dl_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)

        # Validation (using test set for simplicity)
        dl_model.eval()
        with torch.no_grad():
            val_outputs, _ = dl_model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor).item()
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(dl_model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load the best model
    dl_model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate the model
    dl_model.eval()
    with torch.no_grad():
        y_pred_tensor, attention_weights = dl_model(X_test_tensor)
        y_pred = y_pred_tensor.numpy()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Deep Learning Model Evaluation:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2 Score: {r2:.2f}")

    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()

    # Plot attention weights (example for a single sample)
    sample_idx = 0

    print("Shape of attention_weights:", attention_weights.shape)
    #print("Attention weights content:\n", attention_weights)

    # Assuming attention_weights.shape is [batch_size, num_heads, seq_len, seq_len]
    # and we want to average across heads for the first sample
    attention_weights_sample = attention_weights[sample_idx].mean(dim=0).detach().cpu().numpy()

    # No need for squeeze() if the shape is already [seq_len, seq_len]

    if attention_weights_sample.ndim == 2:
        plt.figure(figsize=(8, 5))
        plt.matshow(attention_weights_sample, cmap='viridis')
        plt.xlabel("Features")
        plt.ylabel("Features")
        plt.title(f"Attention Weights for Sample {sample_idx}")
        plt.colorbar()
        plt.show()
    else:
        print("Warning: Cannot plot attention weights due to incorrect shape:", attention_weights_sample.shape)

    return dl_model