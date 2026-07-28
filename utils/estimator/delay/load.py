import torch
from collections import deque
from .lstm import LSTM
from .data import (
    load_tp, 
    create_dataset 
)

# 1. Define constants to match your training script
SPEED = 0
PAYLOAD_SIZE = 19176
LOOKBACK = 5  # Ensure this matches what you trained with!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load the updated delay model
model = LSTM()
model.load_state_dict(torch.load(f"models/model_delay_speed_{SPEED}.pt", map_location=device, weights_only=True))
model.to(device)
model.eval() # Set to evaluation mode

# 3. Load data
tpu, tpd = load_tp(speed=SPEED) 

train_size = int(len(tpd) * 0.67)
train, test = tpd[:train_size], tpd[train_size:]

# 4. Create dataset (passing the required payload_size)
X_train, y_train = create_dataset(train, lookback=LOOKBACK, payload_size=PAYLOAD_SIZE)
X_test, y_test = create_dataset(test, lookback=LOOKBACK, payload_size=PAYLOAD_SIZE)

# 5. Create a dummy sliding window simulating a constant 100 Mbps throughput
s = deque(LOOKBACK * [100.0], LOOKBACK)
l = list(s)

# Reshape dummy window to (batch_size=1, seq_len=5, features=1)
window = torch.tensor(l, dtype=torch.float32).view(1, LOOKBACK, 1).to(device)

print("--- Real Data Sample ---")
print("X_test[0] original shape:", X_test[0].shape)

# Add a batch dimension to X_test[0] using unsqueeze(0)
real_sample = X_test[0].unsqueeze(0).to(device)
print("X_test[0] batched shape:", real_sample.shape)

with torch.no_grad():
    # Because we fixed lstm.py, we just grab the item directly
    print(f"Prediction (Real): {model(real_sample).item():.4f} seconds")
    print(f"Ground Truth (Real): {y_test[0].item():.4f} seconds")

print("\n--- Dummy Window Sample (Constant 100 Mbps) ---")
print("Window batched shape:", window.shape)
print("Window raw values:", window.view(-1).tolist())

with torch.no_grad():
    print(f"Prediction (Dummy): {model(window).item():.4f} seconds")