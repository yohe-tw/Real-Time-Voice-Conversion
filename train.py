import torch
import torchaudio
import torchaudio.transforms
import torchaudio.functional
import torch.optim as optim
import torch
import os
from torch.utils.data import DataLoader, random_split
import sys
from tqdm import tqdm
from utils.utils import parse_wav_files
from utils.data import VocalTechniqueDataset
from utils.loss import SobelEdgeLoss
from unet.unet_2 import UNet
import argparse

# Ensure device is set
device = "cuda" if torch.cuda.is_available() else "cpu"


#################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default="./outputmodel")
    parser.add_argument('--dataset_path', type=str, default="..")
    args = parser.parse_args()
    return args

args = parse_args()


base_directory = args.dataset_path # Replace with the actual path
wav_dict = parse_wav_files(base_directory)
os.makedirs(args.output_path, exist_ok=True)

# Remove file paths with "Paired_Speech_Group"
for key, value in wav_dict.items():
    value['file_path'] = [
        path for path in value['file_path']
        if "Paired_Speech_Group" not in path
    ]
#################


# Compute global min and max
# global_mel_min, global_mel_max = compute_global_min_max(wav_dict)
# Min: -100.0, Max: 53.63450622558594
global_mel_min = -100.0
global_mel_max = 54
print(f"Global Mel-Spectrogram Min: {global_mel_min}, Max: {global_mel_max}")


# create dataset
dataset = VocalTechniqueDataset(wav_dict, global_mel_min=global_mel_min, global_mel_max=global_mel_max)
dataset.validate_pairs()


# Define split sizes (e.g., 80% train, 10% validation, 10% test)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Calculate lengths of each split
total_size = len(dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size  # Ensures all data is used

# Perform random split
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Wrap datasets into DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Print dataset sizes
print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")



model = UNet(latent_dim=32).to(device)  # Move to GPU if available
# model = ContextUnet(in_channels=1, n_classes=5).to(device)
criterion = SobelEdgeLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10


for epoch in range(num_epochs):
    
    model.train()  # Set the model to training mode
    epoch_loss = 0.0

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    # Add a progress bar for batches
    with tqdm(total=len(train_loader), desc="Training", unit="batch") as pbar:
        for batch_idx, batch in enumerate(train_loader):
            # Move data to the appropriate device
            control_mel = batch["control_mel"].to(device)
            reference_mel = batch["reference_mel"].to(device)
            technique_mel = batch["technique_mel"].to(device)
            label = batch["label"]
            print(control_mel.shape)
            print(label.shape)
            # Forward pass
            outputs = model(control_mel, label.to(device), 0.1)

            # Compute the loss
            loss = criterion(outputs, technique_mel)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update epoch loss
            epoch_loss += loss.item()

            # Print batch details (optional)
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)  # Increment progress bar

    # Print average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_epoch_loss:.6f}")
    torch.save(model.state_dict(), os.path.join(args.output_path, "cond.pt"))







