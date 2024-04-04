import pandas as pd
import torch
from torch.utils.data import Dataset

from train_decoder_gru import SeqToSeqGru
from utils import train, count_parameters, Logger


class RealDataset(Dataset):
    def __init__(self, start_idx, end_idx, lookback_window_size, prediction_size):
        assert end_idx > start_idx
        assert end_idx - start_idx > lookback_window_size + prediction_size
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.lookback_window_size = lookback_window_size
        self.prediction_size = prediction_size
        self.time_series = self.load_data()
        assert end_idx < self.data_size

    def load_data(self):
        csv_file = './data/data_easier.csv'
        column_name = 'target'
        df = pd.read_csv(csv_file)
        y = torch.from_numpy(df[column_name].to_numpy()).float()
        self.data_size = y.shape[0]
        return y[self.start_idx: self.end_idx]

    def __len__(self):
        return self.end_idx - self.start_idx - self.lookback_window_size - self.prediction_size

    def __getitem__(self, idx):
        signal = self.time_series[idx: idx + self.lookback_window_size + self.prediction_size]
        signal = (signal - torch.mean(signal))
        signal = signal / torch.max(torch.abs(signal))
        encoder_input = signal[:self.lookback_window_size]
        decoder_input = signal[self.lookback_window_size - 1:]
        decoder_targets = torch.roll(decoder_input, -1, dims=0)
        return encoder_input.unsqueeze_(dim=-1), decoder_targets[:-1].unsqueeze_(dim=-1)


if __name__ == '__main__':
    LR = 0.000001
    SIGNAL_SIZE = 110
    LOOKBACK_WINDOW_SIZE = 100
    PREDICTION_SIZE = 10
    assert LOOKBACK_WINDOW_SIZE + PREDICTION_SIZE == SIGNAL_SIZE
    EPOCH_FROM = 0
    EPOCH_TO = 1000
    SEND_TO_WANDB = True

    #### BEGIN: Load model and init Logger
    model = SeqToSeqGru(encoder_input_length=LOOKBACK_WINDOW_SIZE, decoder_input_length=PREDICTION_SIZE, decoder_hidden_dim=140)
    model.load_state_dict(torch.load('./checkpoints/04-Apr-2024_16-50-22_8drs2gg8/17500_epochs'))

    # checkpoint_path = './checkpoints/M3BBAG/02-04-2024_14-41-40/6000_epochs'
    checkpoint_path = None

    hyperparameters = ["amplitude mod", f"LR={LR}", f"PARAMS={count_parameters(model)}", f"SIGNAL_SIZE={SIGNAL_SIZE}",
                       f"LOOKBACK_WINDOW_SIZE={LOOKBACK_WINDOW_SIZE}", f"PREDICTION_SIZE={PREDICTION_SIZE}"]

    if checkpoint_path:
        EPOCH_FROM = int(checkpoint_path.split("/")[-1].split("_")[0])
        run_id = checkpoint_path.split("/")[-3]
        model.load_state_dict(torch.load(checkpoint_path))
        logger = Logger(
            "finetune (3,45)", send_to_wandb=SEND_TO_WANDB, id_resume=run_id,
            hyperparameters=hyperparameters
            )
    else:
        logger = Logger("finetune (3,45)", send_to_wandb=SEND_TO_WANDB, hyperparameters=hyperparameters)
    ### END

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_dataset = RealDataset(start_idx=0, end_idx=500, lookback_window_size=100, prediction_size=10)
    eval_dataset = RealDataset(start_idx=500, end_idx=768 - 1, lookback_window_size=100, prediction_size=10)

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
        save_freq=20
    )
    logger.finish()
