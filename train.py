import torch
from torch.utils.data import Dataset

from run_config import LOOKBACK_WINDOW_SIZE, PREDICTION_SIZE, SIGNAL_SIZE, NUM_COMPONENTS_RANGE, PERIODS_RANGE, NOISE
from utils import generate_signal, count_parameters, Logger, train, \
    generate_signal_with_amplitude_mod


class SeqToSeqDataset(Dataset):
    def __init__(self, size, num_samples, split_idx, num_components, periods_range, noise, return_decoder_input=True):
        assert num_samples > split_idx
        self.size = size
        self.num_samples = num_samples
        self.split_idx = split_idx
        self.periods_range = periods_range
        self.num_components = num_components
        self.noise = noise
        self.return_decoder_input = return_decoder_input

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        _, signal = generate_signal(
            num_samples=self.num_samples, num_components=self.num_components,
            periods_range=self.periods_range, noise=self.noise
        )
        encoder_input = signal[:self.split_idx]
        decoder_input = signal[self.split_idx - 1:]
        decoder_targets = torch.roll(decoder_input, -1, dims=0)
        return encoder_input.unsqueeze_(dim=-1), decoder_targets[:-1].unsqueeze_(dim=-1)


class MLPEncoder(torch.nn.Module):
    def __init__(self, signal_plus_dft_dim, decoder_hidden_dim) -> None:
        super(MLPEncoder, self).__init__()
        self.relu = torch.nn.LeakyReLU()
        self.linear_last = torch.nn.Linear(signal_plus_dft_dim, decoder_hidden_dim)
        self.linear_1 = torch.nn.Linear(signal_plus_dft_dim, 100)
        self.linear_2 = torch.nn.Linear(100, signal_plus_dft_dim)

    def forward(self, encoder_input):
        """
        Args:
            encoder_input: torch.Tensor[N, L_encoder, 1]

        Returns:
            torch.Tensor: [N, h_encoder + fft]
        """
        fft_result = torch.fft.rfft(encoder_input, dim=1, norm="forward")
        amplitudes = torch.abs(fft_result)  # = [N, f_dim, 1]
        phases = torch.angle(fft_result)
        signal_cat_fft = torch.concat((encoder_input, amplitudes, phases), dim=1)

        x = self.linear_1(torch.squeeze(signal_cat_fft, dim=-1))
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_last(x)  # [N, L + 2*f_dim]
        return x


class GruDecoderCell(torch.nn.Module):
    def __init__(self, hidden_size) -> None:
        super(GruDecoderCell, self).__init__()
        self.gru_cell = torch.nn.GRUCell(input_size=1, hidden_size=hidden_size)

    def forward(self, input, h):
        return self.gru_cell(input, h)


class SeqToSeqGru(torch.nn.Module):
    def __init__(self, encoder_input_length=100, decoder_input_length=50, decoder_hidden_dim=200) -> None:
        super(SeqToSeqGru, self).__init__()
        self.encoder_in_dim = encoder_input_length
        self.decoder_in_dim = decoder_input_length

        self.decoder_hidden_dim = decoder_hidden_dim
        self.signal_plus_dft_dim = self.encoder_in_dim + 2 * (
                int(self.encoder_in_dim / 2) + 1)  # hidden dim + cat 2 * rfft

        self.encoder = MLPEncoder(
            signal_plus_dft_dim=self.signal_plus_dft_dim,
            decoder_hidden_dim=self.decoder_hidden_dim
        )
        self.decoder_cell = GruDecoderCell(hidden_size=self.decoder_hidden_dim)
        self.leaky_relu = torch.nn.LeakyReLU()
        self.linear_1 = torch.nn.Linear(self.decoder_hidden_dim, int(self.decoder_hidden_dim / 2))
        self.linear_2 = torch.nn.Linear(int(self.decoder_hidden_dim / 2), int(self.decoder_hidden_dim / 4))
        self.linear_3 = torch.nn.Linear(int(self.decoder_hidden_dim / 4), 1)

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
        mlp_outputs = torch.empty((encoder_input.shape[0], self.decoder_in_dim, 1), dtype=torch.float)  # [N, dec_in, 1]
        # No teacher forcing: autoregress
        # init the autoregression with the last value in the lookback window
        decoder_input = encoder_input[:, -1, :]  #
        for i in range(self.decoder_in_dim):
            h_decoder = self.decoder_cell(decoder_input, h_decoder)
            x = self.linear_1(h_decoder)
            x = self.leaky_relu(x)
            x = self.linear_2(x)
            x = self.leaky_relu(x)
            x = self.linear_3(x)
            decoder_input = x.clone().detach()
            mlp_outputs[:, i, :] = x
        return mlp_outputs


if __name__ == '__main__':
    LR = 0.0001
    EPOCH_FROM = 9000
    EPOCH_TO = 18000
    SEND_TO_WANDB = True

    #### BEGIN: Load model and init Logger
    model = SeqToSeqGru(
        encoder_input_length=LOOKBACK_WINDOW_SIZE, decoder_input_length=PREDICTION_SIZE,
        decoder_hidden_dim=140
    )

    resume_path = 'checkpoints/04-Apr-2024_15-51-32_8drs2gg8/9000_epochs'
    # resume_path = None

    # load_checkpoint_path = 'checkpoints/04-Apr-2024_14-25-34_r46ycq3s/2600_epochs'
    load_checkpoint_path = None

    hyperparameters = ["no amplitude mod, no pre training", f"LR={LR}", f"PARAMS={count_parameters(model)}",
                       f"SIGNAL_SIZE={SIGNAL_SIZE}", f"LOOKBACK_WINDOW_SIZE={LOOKBACK_WINDOW_SIZE}",
                       f"PREDICTION_SIZE={PREDICTION_SIZE}", f"NUM_COMPS={NUM_COMPONENTS_RANGE}",
                       f"PERIODS_RANGE={PERIODS_RANGE}", f"NOISE={NOISE}"]

    if resume_path:
        EPOCH_FROM = int(resume_path.split("/")[-1].split("_")[0])
        run_id = resume_path.split("/")[-2].split("_")[-1]
        model.load_state_dict(torch.load(resume_path))
        logger = Logger(
            "decoder-only-lstm", send_to_wandb=SEND_TO_WANDB, id_resume=run_id,
            hyperparameters=hyperparameters
        )
    else:
        if load_checkpoint_path:
            model.load_state_dict(torch.load(load_checkpoint_path))
        logger = Logger("(3,45) on (3,10)", send_to_wandb=SEND_TO_WANDB, hyperparameters=hyperparameters)


    ### END

    def loss_function(pred, labels):
        weights = 1. / (torch.arange(pred.shape[1]) + 1)
        return torch.mean(weights @ (pred - labels) ** 2)


    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_dataset = SeqToSeqDataset(
        size=1000, num_samples=SIGNAL_SIZE, split_idx=LOOKBACK_WINDOW_SIZE,
        num_components=NUM_COMPONENTS_RANGE, periods_range=PERIODS_RANGE, noise=NOISE,
        return_decoder_input=True
    )
    eval_dataset = SeqToSeqDataset(
        size=1000, num_samples=SIGNAL_SIZE, split_idx=LOOKBACK_WINDOW_SIZE,
        num_components=NUM_COMPONENTS_RANGE, periods_range=PERIODS_RANGE, noise=NOISE,
        return_decoder_input=False
    )

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
        logger=logger,
        bs=128
    )
    logger.finish()
