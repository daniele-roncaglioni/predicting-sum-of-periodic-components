import torch
from matplotlib import pyplot as plt

from run_config import LOOKBACK_WINDOW_SIZE, PREDICTION_SIZE, SIGNAL_SIZE
from train_decoder_gru import SeqToSeqGru
from utils import plot_predictions
import pandas as pd

model1 = SeqToSeqGru(
    encoder_input_length=LOOKBACK_WINDOW_SIZE, decoder_input_length=PREDICTION_SIZE,
    decoder_hidden_dim=140
)
# model2 = SeqToSeqGru(encoder_input_length=LOOKBACK_WINDOW, decoder_input_length=PREDICTION_SIZE, decoder_hidden_dim=200)
# model1.load_state_dict(torch.load('./checkpoints/wghp05nl/03-04-2024_19-21-28/900_epochs'))
# model2.load_state_dict(torch.load('./checkpoints/faab98ah/03-04-2024_15-46-05/1700_epochs')) #old
# model3.load_state_dict(torch.load('./checkpoints/wghp05nl/03-04-2024_19-21-28/2000_epochs'))
model1.load_state_dict(torch.load('./checkpoints/04-Apr-2024_15-51-32_8drs2gg8/8500_epochs'))
# model2.load_state_dict(torch.load('./checkpoints/6o841cot/03-04-2024_21-49-30/2000_epochs'))


plot_predictions((model1,))

csv_file = './data/data_easier.csv'
column_name = 'target'

# Read the CSV file using pandas
df = pd.read_csv(csv_file)

# Select the column and convert it to a NumPy array
column_data = df[column_name].to_numpy()
avg_loss = 0
# Convert the NumPy array to a PyTorch tensor
for n in range(500, len(column_data) - SIGNAL_SIZE):
    targets = torch.tensor(column_data, dtype=torch.float32)[n:n + SIGNAL_SIZE]  # or torch.float64 for double precision
    targets = (targets - torch.mean(targets))
    targets = targets / torch.max(torch.abs(targets))
    lookback_window = targets[:LOOKBACK_WINDOW_SIZE]
    pred_window = targets[LOOKBACK_WINDOW_SIZE:]
    model_pred = model1(lookback_window.view(1, LOOKBACK_WINDOW_SIZE, 1)).view(SIGNAL_SIZE - LOOKBACK_WINDOW_SIZE).detach()
    avg_loss += torch.mean((model_pred - pred_window) ** 2)
    # plt.plot(range(n, n + SIGNAL_SIZE), targets)
    # plt.plot(range(n + LOOKBACK_WINDOW_SIZE, n + SIGNAL_SIZE), model_pred, '-x')
    # plt.axvspan(n, n + 99, color='grey', alpha=0.3)
    # plt.savefig(f'./predictions/7/{n}.png')
    # plt.close()
print(f"Avg pred loss: {avg_loss / (len(column_data) - 500)} ")
