import torch
from matplotlib import pyplot as plt

from train_decoder_gru import MLPEncoder, GruDecoderCell, SeqToSeqGru
from utils import plot_predictions
import pandas as pd


class SeqToSeqGruOld(torch.nn.Module):
    def __init__(self, encoder_input_length=100, decoder_input_length=50) -> None:
        super(SeqToSeqGruOld, self).__init__()
        self.encoder_in_dim = encoder_input_length
        self.decoder_in_dim = decoder_input_length

        self.decoder_hidden_dim = 100
        self.signal_plus_dft_dim = self.encoder_in_dim + 2 * (
                int(self.encoder_in_dim / 2) + 1)  # hidden dim + cat 2 * rfft

        self.encoder = MLPEncoder(signal_plus_dft_dim=self.signal_plus_dft_dim,
                                  decoder_hidden_dim=self.decoder_hidden_dim)
        self.decoder_cell = GruDecoderCell(hidden_size=self.decoder_hidden_dim)
        self.leaky_relu = torch.nn.LeakyReLU()
        self.linear_1 = torch.nn.Linear(self.decoder_hidden_dim, int(self.decoder_hidden_dim / 2))
        self.linear_2 = torch.nn.Linear(int(self.decoder_hidden_dim / 2), 1)

    def forward(self, encoder_input):
        """
        Args:
            encoder_input: torch.Tensor[N, L_encoder, 1]

        Returns:
            torch.Tensor: [L_decoder, 1]
        """

        # Compress all of lookback window into hidden state with encoder and concat with fft
        h_n_augemented = self.encoder(encoder_input)  # [N, h_encoder + fft]

        # init hidden state for decoder is the compressed lookback window hidden state from the encoder
        h_decoder = h_n_augemented
        mlp_outputs = torch.empty((encoder_input.shape[0], self.decoder_in_dim, 1),
                                  dtype=torch.float)  # [N, dec_in, 1]
        # No teacher forcing: autoregress
        # init the autoregression with the last value in the lookback window
        decoder_input = encoder_input[:, -1, :]  #
        for i in range(self.decoder_in_dim):
            h_decoder = self.decoder_cell(decoder_input, h_decoder)
            # decoder_outputs[:, i, :] = h_decoder
            x = self.linear_1(h_decoder)
            x = self.leaky_relu(x)
            x = self.linear_2(x)
            decoder_input = x.clone().detach()
            mlp_outputs[:, i, :] = x
        return mlp_outputs


model = SeqToSeqGru(encoder_input_length=100, decoder_input_length=20)
# model.load_state_dict(torch.load('./checkpoints/xq3vrloq/03-04-2024_14-28-07/400_epochs'))
model.load_state_dict(torch.load('./checkpoints/mbrueco8/03-04-2024_11-42-54/2200_epochs'))

plot_predictions(model, 120, 100, noise=True)
"""
csv_file = './data/data_easier.csv'
column_name = 'target'

# Read the CSV file using pandas
df = pd.read_csv(csv_file)

# Select the column and convert it to a NumPy array
column_data = df[column_name].to_numpy()

# Convert the NumPy array to a PyTorch tensor
for n in range(500, len(column_data) - 120):
    targets = torch.tensor(column_data, dtype=torch.float32)[n:n + 120]  # or torch.float64 for double precision
    targets = (targets - torch.mean(targets))
    targets = targets / torch.max(torch.abs(targets))
    lookback_window = targets[:100]
    pred_window = targets[100:]
    model_pred = model(lookback_window.view(1, 100, 1)).view(120 - 100).detach()
    plt.plot(range(n, n + 120), targets)
    plt.plot(range(n + 100, n + 120), model_pred, '-x')
    plt.axvspan(n, n + 99, color='grey', alpha=0.3)
    plt.savefig(f'./predictions/2/{n}.png')
    plt.close()
"""
