import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Import our custom architectures
from model_cnn import CNN1DModel
from model_transformer import EEGTransformer
from model_unet import UNet1DModel

# =====================================================================
# 1. Specialized Loss Function: Focal Loss for Extreme Class Imbalance
# =====================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # inputs should be raw logits before sigmoid!
        bce_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-bce_loss)
        
        # Add alpha weighting for positive vs negative class
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# =====================================================================
# 2. Event-Based EEG Dataset (Prevents Data Leakage)
# =====================================================================
class EEGEventDataset(Dataset):
    def __init__(self, file_paths, window_sec=4, overlap_sec=2):
        """
        Takes a list of specific EDF files (e.g. only files 1-20 for a patient)
        to prevent overlapping windows from leaking between train/test splits.
        """
        self.samples = []
        self.labels = []
        
        # Placeholder for data loading logic
        # Here we would call `preprocess_edf` and assign labels based on the summary.txt
        # For demonstration of the training loop structure, we generate mock data tensors.
        print(f"Loading {len(file_paths)} files into memory (simulated)...")
        self.samples = torch.randn(1000, 18, 1024) # 1000 windows, 18 channels, 1024 samples
        self.labels = torch.zeros(1000)
        
        # Simulate 2% seizures
        seizure_indices = np.random.choice(1000, 20, replace=False)
        self.labels[seizure_indices] = 1.0
        
        print(f"Dataset created: {len(self.samples)} windows ({sum(self.labels)} seizures)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# =====================================================================
# 3. Main Training Routine
# =====================================================================
def train_model(model_name, epochs=10, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}\nStarting Training for {model_name.upper()} on {device}\n{'='*50}")

    if model_name == 'cnn':
        model = CNN1DModel().to(device)
    elif model_name == 'transformer':
        model = EEGTransformer().to(device)
    elif model_name == 'unet':
        model = UNet1DModel().to(device)
    else:
        raise ValueError("Invalid model name")

    # The models currently output sigmoid(x) directly. 
    # For numerical stability with Focal Loss, we technically want logits.
    # But since we use BCELoss as a fallback here (assuming sigmoid output), we do:
    criterion = nn.BCELoss() # Or adapt models to output raw logits and use FocalLoss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 1. Hard-split by files to prevent overlapping window leakage
    train_files = ['chb01_01.edf', 'chb01_02.edf', 'chb01_03.edf']
    val_files   = ['chb01_04.edf', 'chb01_05.edf']

    train_dataset = EEGEventDataset(train_files)
    val_dataset   = EEGEventDataset(val_files)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs) # outputs are (Batch,)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_preds = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(predictions.cpu().numpy())
                
        # Calculate proper clinical metrics
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
                
        print(f"Epoch [{epoch:02d}/{epochs}] | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Precision: {precision:.3f} | Recall(Sens): {recall:.3f} | F1: {f1:.3f}")

    print(f"Training for {model_name} completed.")
    # torch.save(model.state_dict(), f"weights_{model_name}.pth")

if __name__ == "__main__":
    train_model('cnn', epochs=5)
    train_model('transformer', epochs=5)
    train_model('unet', epochs=5)
