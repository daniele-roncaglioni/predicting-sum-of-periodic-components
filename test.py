import torch
from matplotlib import pyplot as plt

from train_decoder_gru import SeqToSeqGru
from utils import plot_predictions
import pandas as pd

model = SeqToSeqGru(encoder_input_length=100, decoder_input_length=20)
# model.load_state_dict(torch.load('./checkpoints/69ua59z5/02-04-2024_23-58-32/1100_epochs'))
model.load_state_dict(torch.load('./checkpoints/55ma169j/02-04-2024_17-35-11/1000_epochs'))

# plot_predictions(model, 120, 100, noise=True)

csv_file = './data/data_easier.csv'
column_name = 'target'

# Read the CSV file using pandas
df = pd.read_csv(csv_file)

# Select the column and convert it to a NumPy array
column_data = df[column_name].to_numpy()

# Convert the NumPy array to a PyTorch tensor
n = 598
targets = torch.tensor(column_data, dtype=torch.float32)[n:n + 120]  # or torch.float64 for double precision
targets = (targets - torch.mean(targets))
targets = targets / torch.max(torch.abs(targets))
lookback_window = targets[:100]
pred_window = targets[100:]
model_pred = model(lookback_window.view(1, 100, 1)).view(120 - 100).detach()
plt.plot(range(120), targets)
plt.plot(range(100, 120), model_pred, '-x')
plt.axvspan(0, 99, color='grey', alpha=0.3)
plt.show()
