import torch
from torch.utils.data import Dataset

from utils import generate_signal, count_parameters, Logger, train


class SeqToSeqDataset(Dataset):
    def __init__(self, size, num_samples=150, split_idx=100, lookback_window_size=100, return_decoder_input=True):
        assert num_samples > split_idx
        self.size = size
        self.num_samples = num_samples
        self.split_idx = split_idx
        self.lookback_window_size = lookback_window_size
        self.return_decoder_input = return_decoder_input

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        _, signal = generate_signal(num_samples=self.num_samples, periods_range=(2, self.lookback_window_size))
        encoder_input = signal[:self.split_idx]
        decoder_input = signal[self.split_idx - 1:]
        decoder_targets = torch.roll(decoder_input, -1, dims=0)
        if self.return_decoder_input:
            return torch.concat((encoder_input, decoder_input[:-1]), dim=0).unsqueeze_(dim=-1), decoder_targets[
                                                                                                :-1].unsqueeze_(dim=-1)
        else:
            return encoder_input.unsqueeze_(dim=-1), decoder_targets[:-1].unsqueeze_(dim=-1)


class GruEncoder(torch.nn.Module):
    def __init__(self, hidden_size) -> None:
        super(GruEncoder, self).__init__()
        self.gru = torch.nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True, num_layers=1)

    def forward(self, encoder_input):
        """
        Args:
            encoder_input: torch.Tensor[N, L_encoder, 1]

        Returns:
            torch.Tensor: [N, h_encoder + fft]
        """
        _, h_n = self.gru(encoder_input)  # h_n =[1, N, h_dim]
        fft_result = torch.fft.rfft(encoder_input, dim=1, norm="forward")
        amplitudes = torch.abs(fft_result)  # = [N, f_dim, 1]
        phases = torch.angle(fft_result)
        return torch.concat(
            (h_n, amplitudes.view(1, encoder_input.shape[0], -1), phases.view(1, encoder_input.shape[0], -1)),
            dim=-1).squeeze_(dim=0)


class GruDecoderCell(torch.nn.Module):
    def __init__(self, hidden_size) -> None:
        super(GruDecoderCell, self).__init__()
        self.gru_cell = torch.nn.GRUCell(input_size=1, hidden_size=hidden_size)

    def forward(self, input, h):
        return self.gru_cell(input, h)


class SeqToSeqGru(torch.nn.Module):
    def __init__(self, encoder_input_length=100, decoder_input_length=50) -> None:
        super(SeqToSeqGru, self).__init__()
        self.encoder_in_dim = encoder_input_length
        self.decoder_in_dim = decoder_input_length

        self.encoder_hidden_dim = 10
        self.enocoder_output_dim = self.encoder_hidden_dim + 2 * (
                int(self.encoder_in_dim / 2) + 1)  # hidden dim + cat 2 * rfft

        self.encoder = GruEncoder(hidden_size=self.encoder_hidden_dim)
        self.decoder_cell = GruDecoderCell(hidden_size=self.enocoder_output_dim)
        self.leaky_relu = torch.nn.LeakyReLU()
        self.linear_1 = torch.nn.Linear(self.enocoder_output_dim, int(self.enocoder_output_dim / 2))
        self.linear_2 = torch.nn.Linear(int(self.enocoder_output_dim / 2), 1)

    def forward(self, input):
        """
        Args:
            encoder_input: torch.Tensor[N, L_encoder, 1]
            decoder_input: torch.Tensor[N, L_decoder, 1] | None

        Returns:
            torch.Tensor: [L_decoder, 1]
        """
        if input.shape[1] == self.encoder_in_dim:
            encoder_input = input
            decoder_input = None
        elif input.shape[1] == self.encoder_in_dim + self.decoder_in_dim:
            encoder_input, decoder_input = input[:, :self.encoder_in_dim, :], input[:, self.encoder_in_dim:, :]
        else:
            raise "input shape mismatch"
        # assert encoder_input.shape[1] == self.encoder_in_dim
        # assert decoder_input.shape[1] == self.decoder_in_dim

        # Compress all of lookback window into hidden state with encoder and concat with fft
        h_n_augemented = self.encoder(encoder_input)  # [N, h_encoder + fft]

        # init hidden state for decoder is the compressed lookback window hidden state from the encoder
        h_decoder = h_n_augemented
        decoder_outputs = torch.empty((encoder_input.shape[0], self.decoder_in_dim, h_n_augemented.shape[-1]),
                                      dtype=torch.float)  # [N, dec_in, h_enc+fft]
        mlp_outputs = torch.empty((encoder_input.shape[0], self.decoder_in_dim, 1),
                                  dtype=torch.float)  # [N, dec_in, 1]
        # teacher forcing: feed actual sequence to decoder
        if decoder_input is not None:  # Loop over inputs and feed them to decoder cell while updating the hidden state
            for i in range(self.decoder_in_dim):
                input_element = decoder_input[:, i, :]
                h_decoder = self.decoder_cell(input_element, h_decoder)  # [N, h_enc + fft]
                decoder_outputs[:, i, :] = h_decoder
            x = self.linear_1(decoder_outputs)
            x = self.leaky_relu(x)
            x = self.linear_2(x)
            return x
        else:  # No teacher forcing: autoregress
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


if __name__ == '__main__':
    LR = 0.002
    SIGNAL_SIZE = 105
    LOOKBACK_WINDOW_SIZE = 100
    PREDICTION_SIZE = 5
    assert LOOKBACK_WINDOW_SIZE + PREDICTION_SIZE == SIGNAL_SIZE
    EPOCH_FROM = 6000
    EPOCH_TO = 7000
    SEND_TO_WANDB = True

    #### BEGIN: Load model and init Logger
    model = SeqToSeqGru(encoder_input_length=LOOKBACK_WINDOW_SIZE, decoder_input_length=PREDICTION_SIZE)

    checkpoint_path = './checkpoints/M3BBAG/02-04-2024_14-41-40/6000_epochs'
    # checkpoint_path = None

    hyperparameters = [f"LR={LR}", f"PARAMS={count_parameters(model)}", f"SIGNAL_SIZE={SIGNAL_SIZE}",
                       f"LOOKBACK_WINDOW_SIZE={LOOKBACK_WINDOW_SIZE}", f"PREDICTION_SIZE={PREDICTION_SIZE}"]

    if checkpoint_path:
        EPOCH_FROM = int(checkpoint_path.split("/")[-1].split("_")[0])
        run_id = checkpoint_path.split("/")[-3]
        model.load_state_dict(torch.load(checkpoint_path))
        logger = Logger("enocder-decoder-lstm", send_to_wandb=SEND_TO_WANDB, id_resume=run_id,
                        hyperparameters=hyperparameters)
    else:
        logger = Logger("enocder-decoder-lstm", send_to_wandb=SEND_TO_WANDB, hyperparameters=hyperparameters)
    ### END

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_dataset = SeqToSeqDataset(size=1000, num_samples=SIGNAL_SIZE, split_idx=LOOKBACK_WINDOW_SIZE,
                                    return_decoder_input=True)
    eval_dataset = SeqToSeqDataset(size=1000, num_samples=SIGNAL_SIZE, split_idx=LOOKBACK_WINDOW_SIZE,
                                   return_decoder_input=False)

    assert EPOCH_TO > EPOCH_FROM
    print(f"Training from {EPOCH_FROM} to {EPOCH_TO}")

    train(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizer=optimizer,
        epochs_from=EPOCH_FROM,
        epochs_to=EPOCH_TO,
        loss_function=loss_function,
        logger=logger
    )
    logger.finish()
