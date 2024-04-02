import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import wandb
from pathlib import Path
import json
from typing import List
import random
import string
from datetime import datetime


# np.random.seed(0)  # Seed for reproducibility


def generate_signal_with_amplitude_mod(num_samples=118, noise=False, num_components_range=(3, 7),
                                       periods_range=(2, 100)):
    """
    Generates a signal with dynamic frequency and amplitude drift.

    :param num_samples: Total number of samples in the signal.
    :param noise: Whether to add Gaussian noise to the signal.
    :param num_components_range: Tuple indicating the range of number of sinusoidal components to generate.
    :param periods_range: Tuple indicating the range of period lengths for the sinusoidal components.
    :return: Sample indices and the generated signal as PyTorch tensors.
    """
    if type(num_components_range) == int:
        num_components_range = (num_components_range, num_components_range + 1)
    else:
        assert num_components_range[0] < num_components_range[1]

    if type(periods_range) == int:
        periods_range = (periods_range, periods_range + 1)
    else:
        assert periods_range[0] <= periods_range[1]
    # Validation and setup
    assert num_components_range[0] < num_components_range[1], "Invalid component range"
    assert periods_range[0] <= periods_range[1], "Invalid period range"

    num_components = np.random.randint(*num_components_range)
    periods = np.random.randint(*periods_range, num_components)
    phases = np.random.rand(num_components) * 2 * np.pi
    amplitudes = np.random.rand(num_components)
    amplitudes /= np.sum(amplitudes)  # Normalize amplitudes

    samples = np.arange(num_samples)
    signal = np.zeros(num_samples)

    for i in range(num_components):
        m = np.random.randint(1, num_samples - 1)  # Midpoint for drift
        r1 = np.random.uniform(0.5, 2)  # Random drift factors

        amp_adjust = np.linspace(1, r1, m)
        amp_adjust = np.concatenate([amp_adjust, np.linspace(r1, 1, num_samples - m)])

        period_samples = (1 / periods[i])
        amplitude_samples = amplitudes[i] * amp_adjust  # Apply amplitude drift

        component_signal = amplitude_samples * np.sin(2 * np.pi * period_samples * samples + phases[i])
        signal += component_signal

    # Normalize signal to [-1, 1]
    signal = signal / np.max(np.abs(signal))

    # Add random noise
    if noise:
        noise_level = np.random.normal(0, 0.1, signal.shape)
        signal += noise_level
        signal = signal / np.max(np.abs(signal))  # Normalize signal again

    return torch.from_numpy(samples).float(), torch.from_numpy(signal).float()


def generate_signal(num_samples=118, noise=False, num_components=(3, 7), periods_range=(2, 100)):
    """
    :param num_samples: Total number of samples in the signal
    :return: 
    """
    if type(num_components) == int:
        num_components = (num_components, num_components + 1)
    else:
        assert num_components[0] < num_components[1]
    if type(periods_range) == int:
        periods_range = (periods_range, periods_range + 1)
    else:
        assert periods_range[0] <= periods_range[1]

    # Randomly choose how many periods to combine
    num_components = np.random.randint(*num_components)

    # Period lengths in samples
    periods = np.random.randint(*periods_range, num_components)
    phases = np.random.rand(num_components) * 2 * np.pi
    amplitudes = np.random.rand(num_components)
    amplitudes /= np.sum(amplitudes)

    # Sample indices
    samples = np.arange(num_samples)

    # Generate a random continuous periodic signal
    signal = sum(amplitude * np.sin(2 * np.pi * (1 / period) * samples + phase)
                 for amplitude, period, phase in zip(amplitudes, periods, phases))
    signal = signal / np.max(np.abs(signal))  # Normalize signal

    # Add random noise to the signal
    if noise:
        noise = np.random.normal(0, 0.1, signal.shape)
        signal += noise
        signal = signal / np.max(np.abs(signal))  # Normalize signal again

    return torch.from_numpy(samples).to(torch.float), torch.from_numpy(signal).to(torch.float)


def get_fft_harmonics(orignal_signal, analysis_samples, hamming_smoothing=False):
    if hamming_smoothing:
        signal_window = orignal_signal[:analysis_samples]
        window = torch.hamming_window(analysis_samples, periodic=False)
        signal_window = signal_window * window
        fft_result = torch.fft.rfft(signal_window)
    else:
        fft_result = torch.fft.rfft(orignal_signal[:analysis_samples])
    amplitudes = torch.abs(fft_result) / analysis_samples
    phases = torch.angle(fft_result)

    # Double the amplitudes for non-DC components
    # Note: The last component should not be doubled if N is even and represents the Nyquist frequency
    if analysis_samples % 2 == 0:
        # If the original signal length is even, don't double the last component (Nyquist frequency)
        amplitudes[1: -1] *= 2
    else:
        # If the original signal length is odd, all components except the DC can be doubled
        amplitudes[1:] *= 2
    return amplitudes, phases


def get_signal_from_harmonics(amplitudes, phases, num_samples):
    analysis_samples = len(amplitudes) * 2 - 2  # Adjust for rfft output length
    reconstructed_signal = torch.zeros(num_samples, dtype=torch.complex64)
    for index, (amplitude, phase) in enumerate(zip(amplitudes, phases)):
        reconstructed_signal += amplitude * \
                                torch.exp(1j * (2 * torch.pi * index *
                                                torch.arange(num_samples) / analysis_samples + phase))

    # Return the real part of the reconstructed signal
    return reconstructed_signal.real


def reconstruct_signal_fft(orignal_signal, analysis_samples, hamming_smoothing=False):
    # Perform FFT on the entire signal
    amplitudes, phases = get_fft_harmonics(
        orignal_signal, analysis_samples, hamming_smoothing)

    # Return the real part of the reconstructed signal
    return get_signal_from_harmonics(amplitudes, phases, len(orignal_signal))


def plot_signal_and_fft(signal: torch.Tensor, train_test_split_idx: int, hamming_smoothing: bool = False):
    assert (W := signal.shape[0]) >= train_test_split_idx
    t_full = torch.arange(W)

    amplitudes, phases = get_fft_harmonics(
        signal, train_test_split_idx, hamming_smoothing)
    frequency_bins = np.arange(len(amplitudes), dtype=np.float16)

    plt.figure(figsize=(15, 3))

    plt.subplot(1, 3, 1)
    plt.plot(t_full, signal, label="original")
    plt.plot(t_full, get_signal_from_harmonics(
        amplitudes, phases, W), '-x', label="fft reconstruction")
    plt.axvspan(0, train_test_split_idx - 1, color='grey', alpha=0.3)
    plt.xlabel('t [samples]')
    plt.legend(loc='best')
    plt.title('Signal')

    fft_frequencies = frequency_bins / train_test_split_idx
    plt.subplot(1, 3, 2)
    plt.xscale('log')
    plt.plot(fft_frequencies, amplitudes)
    plt.xlabel('Frequency [1/samples]')
    plt.title('Amplitudes')

    fft_periods = frequency_bins.copy()
    fft_periods[1:] = train_test_split_idx / fft_periods[1:]
    plt.subplot(1, 3, 3)
    plt.plot(frequency_bins, amplitudes)
    top_5_indices = np.argsort(amplitudes)[-4:]
    top_5_indices[0] = 1
    plt.xscale('log')
    plt.xticks(frequency_bins[top_5_indices],
               fft_periods[top_5_indices], rotation=70)
    plt.xlabel('T [samples]')
    plt.title('Amplitudes')

    plt.tight_layout()
    plt.show()


def plot_predictions(model, SIGNAL_SIZE, LOOKBACK_WINDOW_SIZE, noise=False):
    def lstm_pred(signal, N):
        signal = torch.clone(signal)
        pred = model(signal[:N].view(1, N, 1))
        return pred.view(SIGNAL_SIZE - N).detach()

    predictions_amount = 10
    fig, axes = plt.subplots(predictions_amount, figsize=(15, 15))
    for ax in axes:
        # t, signal = generate_signal(num_samples=SIGNAL_SIZE, periods_range=(2, 100), noise=noise)
        t, signal = generate_signal_with_amplitude_mod(num_samples=SIGNAL_SIZE, periods_range=(2, 100), noise=noise)

        ax.plot(t, signal, label='input')
        ax.plot(range(LOOKBACK_WINDOW_SIZE, SIGNAL_SIZE), lstm_pred(signal, LOOKBACK_WINDOW_SIZE), '-x',
                label='lstm predicted')
        ax.legend(loc='best')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Amplitude')

        ax.axvspan(0, LOOKBACK_WINDOW_SIZE - 1, color='grey', alpha=0.3)

    fig.tight_layout()
    fig.show()


class Logger:
    def __init__(self, run_name, send_to_wandb=False, id_resume=None, hyperparameters: List[str] = []) -> None:

        common_kwargs = {
            'project': "time-series",
            'tags': hyperparameters,
            'name': run_name,
        }
        if send_to_wandb and not id_resume:
            wandb.init(
                **common_kwargs
            )
            self.run_id = wandb.run.id
        if send_to_wandb and id_resume:
            wandb.init(
                **common_kwargs,
                id=id_resume,
                resume=True
            )
            self.run_id = id_resume

        if not send_to_wandb and id_resume:
            self.run_id = id_resume
        elif not send_to_wandb:
            self.run_id = ''.join(random.choices(
                string.ascii_uppercase + string.digits, k=6))

        print("RUN ID", self.run_id)
        self.send_to_wandb = send_to_wandb

        base_dir = Path(f"./checkpoints/{self.run_id}/")
        base_dir.mkdir(parents=True, exist_ok=True)
        self.new_checkpoint_dir = Path(
            f"./checkpoints/{self.run_id}/{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}")
        self.new_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(f"{self.new_checkpoint_dir}/config.json", 'w') as f:
            json.dump({"run_name": run_name, "send_to_wandb": send_to_wandb,
                       "id_resume": self.run_id, "hyperparameters": hyperparameters}, f)

    def log(self, data, step):
        if self.send_to_wandb:
            wandb.log(data, step)

    def finish(self):
        if self.send_to_wandb:
            wandb.finish()


def count_parameters(model):
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def train(
        model,
        train_dataset,
        eval_dataset,
        optimizer,
        loss_function,
        logger,
        epochs_from=0,
        epochs_to=500,
        device='cpu',
        bs=32,
):
    print("DEVICE", device)
    model.to(device)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=bs, shuffle=True)
    for i in range(epochs_from, epochs_to + 1):
        model.train()
        train_loss = 0
        for seq, labels in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(seq.to(device))
            single_loss = loss_function(y_pred, labels.to(device))
            single_loss.backward()
            train_loss += single_loss
            optimizer.step()

        if i % 25 == 0:
            model.eval()
            val_loss = 0
            for seq, labels in eval_dataloader:
                y_pred = model(seq.to(device))
                single_loss = loss_function(y_pred, labels.to(device))
                val_loss += single_loss
            print(f'TRAIN: epoch: {i},  loss: {train_loss.item()}')
            print(f'VAL: epoch: {i},  val loss: {val_loss.item()} \n')
            logger.log({"train loss": train_loss.item(),
                        "val loss": val_loss.item()}, i)
            if i % 100 == 0:
                torch.save(model.state_dict(),
                           f"{logger.new_checkpoint_dir}/{i}_epochs")


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    samples, y = generate_signal_with_amplitude_mod()
    plt.plot(samples, y)
    plt.show()
