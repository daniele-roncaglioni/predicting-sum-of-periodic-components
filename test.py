import torch
from matplotlib import pyplot as plt

SIGNAL_SIZE = 110
LOOKBACK_WINDOW = 100
PREDICTION_SIZE = 10
from train_decoder_gru import MLPEncoder, GruDecoderCell, SeqToSeqGru
from utils import plot_predictions
import pandas as pd

model1 = SeqToSeqGru(encoder_input_length=LOOKBACK_WINDOW, decoder_input_length=PREDICTION_SIZE, decoder_hidden_dim=200)
model2 = SeqToSeqGru(encoder_input_length=LOOKBACK_WINDOW, decoder_input_length=PREDICTION_SIZE, decoder_hidden_dim=200)
# model1.load_state_dict(torch.load('./checkpoints/wghp05nl/03-04-2024_19-21-28/900_epochs'))
# model2.load_state_dict(torch.load('./checkpoints/faab98ah/03-04-2024_15-46-05/1700_epochs')) #old
# model3.load_state_dict(torch.load('./checkpoints/wghp05nl/03-04-2024_19-21-28/2000_epochs'))
model1.load_state_dict(torch.load('./checkpoints/6o841cot/03-04-2024_21-49-30/1300_epochs'))
model2.load_state_dict(torch.load('./checkpoints/6o841cot/03-04-2024_21-49-30/2000_epochs'))


plot_predictions((model1,model2), SIGNAL_SIZE, LOOKBACK_WINDOW, noise=True)

"""
csv_file = './data/data_easier.csv'
column_name = 'target'

# Read the CSV file using pandas
df = pd.read_csv(csv_file)

# Select the column and convert it to a NumPy array
column_data = df[column_name].to_numpy()

# Convert the NumPy array to a PyTorch tensor
for n in range(500, len(column_data) - SIGNAL_SIZE):
    targets = torch.tensor(column_data, dtype=torch.float32)[n:n + SIGNAL_SIZE]  # or torch.float64 for double precision
    targets = (targets - torch.mean(targets))
    targets = targets / torch.max(torch.abs(targets))
    lookback_window = targets[:LOOKBACK_WINDOW]
    pred_window = targets[LOOKBACK_WINDOW:]
    model_pred = model(lookback_window.view(1, LOOKBACK_WINDOW, 1)).view(SIGNAL_SIZE - LOOKBACK_WINDOW).detach()
    plt.plot(range(n, n + SIGNAL_SIZE), targets)
    plt.plot(range(n + LOOKBACK_WINDOW, n + SIGNAL_SIZE), model_pred, '-x')
    plt.axvspan(n, n + 99, color='grey', alpha=0.3)
    plt.savefig(f'./predictions/3/{n}.png')
    plt.close()
"""

