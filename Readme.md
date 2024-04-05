# Predicting a signal composed of a finite set of harmonics

The goal of this small project was to see how easy or difficult it is to predict a signal made up of a significant amount of harmonics. I randomly generate a set of amplitudes, periods and phases, sum them together to obtain a synthetic signal. I then train a model to predict the composite signal. Obviously training a NN to predict one component is easy, but how about a signal with 40-ish components?

The task is less trivial than it might sound. While on first glance the Discrete Fourier Transform (DFT) might seem like the straightforward solution, it has a key problem. For it to perfectly reconstruct the signal also outside of the window on which you do the DFT (lookback window)  all the harmonics need to have a period that complete an integer number of cycles within it. If they are not, the DFT will give you amplitudes for a Fourier Series that perfectly matches the signal inside the lookback window, but might fail completely outside of it as the signal within the lookback window is just periodically repeated outside of it.

Note: 
- We'll use torch for our Neural Nets.
- We'll set the sample rate `fs=1`
- We'll deal with real signals only, so we stick to the Real DFT.

## DFT

First lets make sure to familiarize with the DFT so that we know how to use it properly. Training a [Neural Network learn to perform a DFT](https://gist.github.com/endolith/98863221204541bf017b6cae71cb0a89) is absolutely possible, but highly inefficient, so we'll provide the DFT as an input to our Network. 

### Theory recap

**1) FFT stands for Fast Fourier Transform and is the `O(nlog(n))` algorithm typically used to compute the DFT.**

Hence torch.fft numpy.fft, scipy.fft or torch.rfft for the Real DFT. 

**2) The DFT is a linear operation that maps a signal of N samples onto N (complex) values which represent the amplitudes and phases of the harmonics/components that make up the signal.**

The frequencies of these components are determined by `N` and the sample rate `fs`: `f[k]=k*fs/N`, or simply `f[k]=k/N` in our case.

**3) While there are N components in the DFT, these only correspond to about `N/2` different and unique fundamental frequencies.**

That is because the DFT maps the signal onto a given positive frequency and its negative separately. The two are functionally equivalent when dealing with real valued signals, so the two corresponding amplitudes are added together, and we just work with the positive frequency.

Specifically we have,

if N is even:
- `N/2`  positive frequency components, including the DC component (the constant value/offset of the signal). 
- `1` value for the Nyquist component `fs/2` (the highest possible frequency) 
- So `N/2+1` in total.

The frequencies will be `f = [0, 1/N, 2/N, ..., 1/2]`

The periods will be     `T = [0, N, N/2, ..., 2]`

if N is odd:
- `(N+1)/2`  positive frequency components including the DC component.
-  Note that we don't reach the Nyquist component in this case, the highest frequency component in the DFT will be slightly below it.

The frequencies will be `f = [0, 1/N, 2/N, ..., 1/2 - 1/2N]`

The periods will be     `T = [0, N, N/2, ..., 2N/(N-1)]`


Let's explore the above in torch and see if we get what we expect.


```python
import torch
# Even N
print("EVEN")
N = 16
expected_components = N/2 + 1
y = torch.arange(N)
dft = torch.fft.rfft(y)
assert int(expected_components) == len(dft)
print(len(dft), expected_components, dft.dtype)

f = torch.fft.rfftfreq(N)
T = 1/torch.fft.rfftfreq(N)
T[0]=0
print(f)
print(T)

# Odd N
print("ODD")
N=15
expected_components = (N+1)/2
y = torch.arange(N)
dft = torch.fft.rfft(y)
assert int(expected_components) == len(dft)
print(len(dft), expected_components, dft.dtype)

f = torch.fft.rfftfreq(N)
T = 1/torch.fft.rfftfreq(N)
T[0]=0
print(f)
print(T)

```

    EVEN
    9 9.0 torch.complex64
    tensor([0.0000, 0.0625, 0.1250, 0.1875, 0.2500, 0.3125, 0.3750, 0.4375, 0.5000])
    tensor([ 0.0000, 16.0000,  8.0000,  5.3333,  4.0000,  3.2000,  2.6667,  2.2857,
             2.0000])
    ODD
    8 8.0 torch.complex64
    tensor([0.0000, 0.0667, 0.1333, 0.2000, 0.2667, 0.3333, 0.4000, 0.4667])
    tensor([ 0.0000, 15.0000,  7.5000,  5.0000,  3.7500,  3.0000,  2.5000,  2.1429])


To get the corresponding amplitudes we do:


```python
amplitudes = torch.abs(dft) / N # Scale by N to retrieve the actual valued of the amplitudes of the harmonics
phases = torch.angle(dft)
print(len(amplitudes), amplitudes)
print(len(phases), phases)
```

    8 tensor([7.0000, 2.4049, 1.2293, 0.8507, 0.6728, 0.5774, 0.5257, 0.5028])
    8 tensor([0.0000, 1.7802, 1.9897, 2.1991, 2.4086, 2.6180, 2.8274, 3.0369])


### Exploration: DFT in pratice

Let's familiarize with how the DFT works in practice.

While obtaining values for the amplitudes, phases, frequencies etc...is interesting it does not help much with understanding.
What is more helpful is to understand the opposite direction of how the frequencies come together to build the signal and predict the signal beyond the sampling window.

So we will do `signal -> dft -> mess with the dft -> see how well the signal is reconstructed inside and outside the sampling window. `

Let's define some helper functions.


```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import wandb
from pathlib import Path

np.random.seed(0)  # Seed for reproducibility

def generate_signal(num_samples=118, noise=False, num_components=(3, 7), periods_range=(2, 100)):
    """
    :param num_samples: Total number of samples in the signal
    :return: 
    """        
    if type(num_components) == int:
        num_components = (num_components, num_components+1)
    else:
        assert num_components[0] < num_components[1]
    if type(periods_range) == int:
        periods_range = (periods_range, periods_range+1)
    else:
        assert periods_range[0] <= periods_range[1]

    num_components = np.random.randint(*num_components)  # Randomly choose how many periods to combine
    
    periods = np.random.randint(*periods_range, num_components)  # Period lengths in samples
    phases = np.random.rand(num_components) * 2 * np.pi
    amplitudes = np.random.rand(num_components)
    amplitudes /= np.sum(amplitudes)

    # Sample indices
    samples = np.arange(num_samples)

    # Generate a random continuous periodic signal
    signal = sum(amplitude * np.sin(2 * np.pi * (1 / period) * samples + phase) for amplitude, period, phase in zip(amplitudes, periods, phases))
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
        reconstructed_signal += amplitude * torch.exp(1j * (2 * torch.pi * index * torch.arange(num_samples) / analysis_samples + phase))

    # Return the real part of the reconstructed signal
    return reconstructed_signal.real


def reconstruct_signal_fft(orignal_signal, analysis_samples, hamming_smoothing=False):
    # Perform FFT on the entire signal
    amplitudes, phases = get_fft_harmonics(orignal_signal, analysis_samples, hamming_smoothing)

    # Return the real part of the reconstructed signal
    return get_signal_from_harmonics(amplitudes, phases, len(orignal_signal))

```

We can use `generate_signal` to produce a random signal or we can also use it to get a non-random output for just one compoent with a specified periodicity.


```python
plt.figure(figsize=(15, 2))

plt.subplot(2, 1, 1)
# signal with 1 component with a period of 10 samples
x,y  = generate_signal(num_components=1, periods_range=10)
plt.plot(x,y)
plt.subplot(2, 1, 2)
# signal with a random amount of components and periods
x,y  = generate_signal()
plt.plot(x,y)

plt.tight_layout()
plt.show()


```


    
![png](assets/readme/output_9_0.png)
    


Let's now define a function that takes care of plotting the original signal, the reconstructed signal and the amplitudes and phases of the harmonics.


```python

def plot_signal_and_fft(signal: torch.Tensor, train_test_split_idx: int, hamming_smoothing: bool = False):
    assert (W := signal.shape[0]) >= train_test_split_idx
    t_full = torch.arange(W)


    amplitudes, phases = get_fft_harmonics(signal, train_test_split_idx, hamming_smoothing)
    frequency_bins = np.arange(len(amplitudes), dtype=np.float16)

    plt.figure(figsize=(15, 3))

    plt.subplot(1, 3, 1)
    plt.plot(t_full, signal, label="original")
    plt.plot(t_full, get_signal_from_harmonics(amplitudes, phases, W), '-x', label="fft reconstruction")
    plt.axvspan(0, train_test_split_idx-1, color='grey', alpha=0.3)
    plt.xlabel('t [samples]')
    plt.legend(loc='best')
    plt.title('Signal')
    
    fft_frequencies = frequency_bins/train_test_split_idx
    plt.subplot(1, 3, 2)
    plt.xscale('log')
    plt.plot(fft_frequencies, amplitudes)
    plt.xlabel('Frequency [1/samples]')
    plt.title('Amplitudes')

    fft_periods = frequency_bins.copy()
    fft_periods[1:]=train_test_split_idx/fft_periods[1:]
    plt.subplot(1, 3, 3)
    plt.plot(frequency_bins, amplitudes)
    top_5_indices = np.argsort(amplitudes)[-4:]
    top_5_indices[0]=1
    plt.xscale('log')
    plt.xticks(frequency_bins[top_5_indices],fft_periods[top_5_indices], rotation=70)
    plt.xlabel('T [samples]')
    plt.title('Amplitudes')


    plt.tight_layout()
    plt.show()


```

Now let's start applying the DFT to 1 component and see how we do.
Let's generate 200 samples and do the DFT on the first 100.


```python
_, y = generate_signal(num_components=1, periods_range=20)
plot_signal_and_fft(y, 100)
```


    
![png](assets/readme/output_13_0.png)
    


Perfection! The DFT correctyl detects that we have 1 harmonic with a period of 20 samples, and our reconstruction then of course is also spot on outside of th fist 100 samples we did the DFT on.

We tried on 100 samples, then 110 samples must be even better! Let's try.


```python
_, y = generate_signal(num_components=1, periods_range=20)
plot_signal_and_fft(y, 110)
```


    
![png](assets/readme/output_15_0.png)
    


Ouch that is not what one could have hoped for. Why?

**Because the DFT "assumes" that all components have a periodicity that completes an integer number of cycles within the sampling window.**

But a sinusoid with a period of 20 does not complete an integer number of cycles in 120 samples. The consequence is that the power of the signal gets spread out a bit over more frequencies (amplitude peak is not perfectly narrow on 20). This is known as spectral leakage.

Moreover our "prediction" of the signal outside the sampling window becomes terrible, as the reconstructed signal is just a copy of  the signal inside the sampling window. This is exactly what the DFT assumes: that the sample window we provide is exactly the period of the signal (up to an integer factor)!

One way to reduce spectral leakage is having smoothing of the signal at the edges of the sampling window towards 0. 


```python
_, y = generate_signal(num_components=1, periods_range=20)
plot_signal_and_fft(y, 110, hamming_smoothing=True)
```


    
![png](assets/readme/output_17_0.png)
    


This localizes the amplitude peak a bit more, but it certianly doesn't help with the predicitve qualities of out reconstruction

Before we move on let's just look at the DFT of a more complex signal.


```python
_, y = generate_signal(200, noise=True)
plot_signal_and_fft(y, 110)
```


    
![png](assets/readme/output_19_0.png)
    


## LSTM

We have looked at how a DFT alone can help us in predicting our signals and the answer is, not very well.
So let's see how a very simple LSTM performs.

Let's define the Dataset class that will feed the data to our LSTM during traing and eval.


```python
import json
from typing import List
import random
import string
from datetime import datetime

class Logger:
    def __init__(self, run_name, send_to_wandb=False, id_resume=None, hyperparameters:List[str]=[]) -> None:
       
        common_kwargs={
            'project':"time-series",
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
        else:
            self.run_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

        print("RUN ID", self.run_id)
        self.send_to_wandb = send_to_wandb
        
        base_dir = Path(f"./checkpoints/{self.run_id}/")
        base_dir.mkdir(parents=True, exist_ok=True)
        self.new_checkpoint_dir = Path(f"./checkpoints/{self.run_id}/{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}")
        self.new_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(f"{self.new_checkpoint_dir}/config.json", 'w') as f:
            json.dump({"run_name": run_name, "send_to_wandb": send_to_wandb, "id_resume": self.run_id, "hyperparameters": hyperparameters}, f)

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
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=bs, shuffle=True)
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
            logger.log({"train loss": train_loss.item(), "val loss": val_loss.item()}, i)
            if i % 100 == 0:
                torch.save(model.state_dict(), f"{logger.new_checkpoint_dir}/{i}_epochs")
        

```

### Small and simple LSTM


```python
class TimeSeriesDataset(Dataset):
    def __init__(self, size, num_samples):
        """
        :param size: Number of samples in the dataset
        """
        self.size = size
        self.num_samples = num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        _, signal = generate_signal(num_samples=self.num_samples)
        target = torch.roll(signal, -1, dims=0)
        return signal.unsqueeze_(dim=-1), target.unsqueeze_(dim=-1)
```

Now let's define the LSTM model itself.


```python
class LSTMPredictor(torch.nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size, batch_first=True, num_layers=1)
        self.linear = torch.nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        x, _ = self.lstm(input_seq)
        x = self.linear(x)
        return x

count_parameters(LSTMPredictor())
```

    +-------------------+------------+
    |      Modules      | Parameters |
    +-------------------+------------+
    | lstm.weight_ih_l0 |    400     |
    | lstm.weight_hh_l0 |   40000    |
    |  lstm.bias_ih_l0  |    400     |
    |  lstm.bias_hh_l0  |    400     |
    |   linear.weight   |    100     |
    |    linear.bias    |     1      |
    +-------------------+------------+
    Total Trainable Params: 41301





    41301



Let's inspect our model architetcure and initialize it.

Now we are ready to go, we just need to initialize the datasets, data loaders, optimizers and start our trainign loop.


```python
LR = 0.002
SIGNAL_SIZE=105
LOOKBACK_WINDOW_SIZE=100
EPOCH_FROM = 0
EPOCH_TO = 3000
SEND_TO_WANDB = True
#### BEGIN: Load model and init Logger
model = LSTMPredictor()

# checkpoint_path = './checkpoints/VVLZNW/1/500_epochs'
checkpoint_path = None

hyperparameters = [f"LR={LR}", f"PARAMS={count_parameters(model)}", f"SIGNAL_SIZE={SIGNAL_SIZE}", f"LOOKBACK_WINDOW_SIZE={LOOKBACK_WINDOW_SIZE}"]

if checkpoint_path:
    EPOCH_FROM = int(checkpoint_path.split("/")[-1].split("_")[0])
    run_id = checkpoint_path.split("/")[-3]
    model.load_state_dict(torch.load(checkpoint_path))
    logger = Logger("small-lstm", send_to_wandb=SEND_TO_WANDB, id_resume=run_id, hyperparameters=hyperparameters)
else:
    logger = Logger("small-lstm", send_to_wandb=SEND_TO_WANDB, hyperparameters=hyperparameters)
### END
    
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
train_dataset = TimeSeriesDataset(size=1000, num_samples=SIGNAL_SIZE) 
eval_dataset = TimeSeriesDataset(size=1000, num_samples=SIGNAL_SIZE)  

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


```

    +-------------------+------------+
    |      Modules      | Parameters |
    +-------------------+------------+
    | lstm.weight_ih_l0 |    400     |
    | lstm.weight_hh_l0 |   40000    |
    |  lstm.bias_ih_l0  |    400     |
    |  lstm.bias_hh_l0  |    400     |
    |   linear.weight   |    100     |
    |    linear.bias    |     1      |
    +-------------------+------------+
    Total Trainable Params: 41301


    Failed to detect the name of this notebook, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable to enable code saving.
    [34m[1mwandb[0m: Currently logged in as: [33mroncaglionidaniele[0m ([33mshape-vs-texture[0m). Use [1m`wandb login --relogin`[0m to force relogin



Tracking run with wandb version 0.16.5



Run data is saved locally in <code>/Users/roncaglionidaniele/Documents/projects/time-series-prediciton/wandb/run-20240402_102819-tju1rve5</code>



Syncing run <strong><a href='https://wandb.ai/shape-vs-texture/time-series/runs/tju1rve5/workspace' target="_blank">small-lstm</a></strong> to <a href='https://wandb.ai/shape-vs-texture/time-series' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/run' target="_blank">docs</a>)<br/>



View project at <a href='https://wandb.ai/shape-vs-texture/time-series' target="_blank">https://wandb.ai/shape-vs-texture/time-series</a>



View run at <a href='https://wandb.ai/shape-vs-texture/time-series/runs/tju1rve5/workspace' target="_blank">https://wandb.ai/shape-vs-texture/time-series/runs/tju1rve5/workspace</a>


    RUN ID 5W53HL
    Training from 0 to 3000
    DEVICE cpu
    TRAIN: epoch: 0,  loss: 4.253148555755615
    VAL: epoch: 0,  val loss: 2.199153184890747 
    
    TRAIN: epoch: 25,  loss: 1.390575885772705
    VAL: epoch: 25,  val loss: 1.3621937036514282 
    
    TRAIN: epoch: 50,  loss: 0.9991466999053955
    VAL: epoch: 50,  val loss: 0.9919925928115845 
    
    TRAIN: epoch: 75,  loss: 0.48536786437034607
    VAL: epoch: 75,  val loss: 0.48534858226776123 
    
    TRAIN: epoch: 100,  loss: 0.32808917760849
    VAL: epoch: 100,  val loss: 0.3068167567253113 
    
    TRAIN: epoch: 125,  loss: 0.3549327850341797
    VAL: epoch: 125,  val loss: 0.38047462701797485 
    
    TRAIN: epoch: 150,  loss: 0.33935683965682983
    VAL: epoch: 150,  val loss: 0.31405001878738403 
    
    TRAIN: epoch: 175,  loss: 0.26200881600379944
    VAL: epoch: 175,  val loss: 0.25902268290519714 
    
    TRAIN: epoch: 200,  loss: 0.24038995802402496
    VAL: epoch: 200,  val loss: 0.2341558337211609 
    
    TRAIN: epoch: 225,  loss: 0.22656749188899994
    VAL: epoch: 225,  val loss: 0.23232676088809967 
    
    TRAIN: epoch: 250,  loss: 0.23107270896434784
    VAL: epoch: 250,  val loss: 0.2327721118927002 
    
    TRAIN: epoch: 275,  loss: 0.21606938540935516
    VAL: epoch: 275,  val loss: 0.2277248501777649 
    
    TRAIN: epoch: 300,  loss: 0.22147917747497559
    VAL: epoch: 300,  val loss: 0.20760945975780487 
    
    TRAIN: epoch: 325,  loss: 0.21088609099388123
    VAL: epoch: 325,  val loss: 0.21732023358345032 
    
    TRAIN: epoch: 350,  loss: 0.20589780807495117
    VAL: epoch: 350,  val loss: 0.20536063611507416 
    
    TRAIN: epoch: 375,  loss: 0.20488175749778748
    VAL: epoch: 375,  val loss: 0.2069343477487564 
    
    TRAIN: epoch: 400,  loss: 0.2091517150402069
    VAL: epoch: 400,  val loss: 0.22765401005744934 
    
    TRAIN: epoch: 425,  loss: 0.2115693837404251
    VAL: epoch: 425,  val loss: 0.21672768890857697 
    
    TRAIN: epoch: 450,  loss: 0.19729548692703247
    VAL: epoch: 450,  val loss: 0.1945153772830963 
    
    TRAIN: epoch: 475,  loss: 0.21520136296749115
    VAL: epoch: 475,  val loss: 0.20812129974365234 
    
    TRAIN: epoch: 500,  loss: 0.19180288910865784
    VAL: epoch: 500,  val loss: 0.20960824191570282 
    
    TRAIN: epoch: 525,  loss: 0.20165011286735535
    VAL: epoch: 525,  val loss: 0.2076336294412613 
    
    TRAIN: epoch: 550,  loss: 0.19163569808006287
    VAL: epoch: 550,  val loss: 0.17862528562545776 
    
    TRAIN: epoch: 575,  loss: 0.19244705140590668
    VAL: epoch: 575,  val loss: 0.20084309577941895 
    
    TRAIN: epoch: 600,  loss: 0.202140212059021
    VAL: epoch: 600,  val loss: 0.1880461424589157 
    
    TRAIN: epoch: 625,  loss: 0.20534123480319977
    VAL: epoch: 625,  val loss: 0.19719472527503967 
    
    TRAIN: epoch: 650,  loss: 0.20530053973197937
    VAL: epoch: 650,  val loss: 0.19188103079795837 
    
    TRAIN: epoch: 675,  loss: 0.1963883638381958
    VAL: epoch: 675,  val loss: 0.20515036582946777 
    
    TRAIN: epoch: 700,  loss: 0.20209914445877075
    VAL: epoch: 700,  val loss: 0.19087639451026917 
    
    TRAIN: epoch: 725,  loss: 0.19317962229251862
    VAL: epoch: 725,  val loss: 0.18497414886951447 
    
    TRAIN: epoch: 750,  loss: 0.20486652851104736
    VAL: epoch: 750,  val loss: 0.19624336063861847 
    
    TRAIN: epoch: 775,  loss: 0.1935804784297943
    VAL: epoch: 775,  val loss: 0.18834711611270905 
    
    TRAIN: epoch: 800,  loss: 0.179626926779747
    VAL: epoch: 800,  val loss: 0.17637808620929718 
    
    TRAIN: epoch: 825,  loss: 0.18028013408184052
    VAL: epoch: 825,  val loss: 0.18196849524974823 
    
    TRAIN: epoch: 850,  loss: 0.18621324002742767
    VAL: epoch: 850,  val loss: 0.17421071231365204 
    
    TRAIN: epoch: 875,  loss: 0.19495409727096558
    VAL: epoch: 875,  val loss: 0.18014511466026306 
    
    TRAIN: epoch: 900,  loss: 0.18251587450504303
    VAL: epoch: 900,  val loss: 0.19391055405139923 
    
    TRAIN: epoch: 925,  loss: 0.1965353786945343
    VAL: epoch: 925,  val loss: 0.1944810003042221 
    
    TRAIN: epoch: 950,  loss: 0.1858014315366745
    VAL: epoch: 950,  val loss: 0.1834276020526886 
    
    TRAIN: epoch: 975,  loss: 0.19137965142726898
    VAL: epoch: 975,  val loss: 0.20726679265499115 
    
    TRAIN: epoch: 1000,  loss: 0.19164501130580902
    VAL: epoch: 1000,  val loss: 0.18424908816814423 
    
    TRAIN: epoch: 1025,  loss: 0.16911888122558594
    VAL: epoch: 1025,  val loss: 0.1917322874069214 
    
    TRAIN: epoch: 1050,  loss: 0.18147757649421692
    VAL: epoch: 1050,  val loss: 0.17264771461486816 
    
    TRAIN: epoch: 1075,  loss: 0.18454135954380035
    VAL: epoch: 1075,  val loss: 0.1834806650876999 
    
    TRAIN: epoch: 1100,  loss: 0.19184142351150513
    VAL: epoch: 1100,  val loss: 0.19093111157417297 
    
    TRAIN: epoch: 1125,  loss: 0.191870778799057
    VAL: epoch: 1125,  val loss: 0.19196611642837524 
    
    TRAIN: epoch: 1150,  loss: 0.19223570823669434
    VAL: epoch: 1150,  val loss: 0.19489766657352448 
    
    TRAIN: epoch: 1175,  loss: 0.18999043107032776
    VAL: epoch: 1175,  val loss: 0.18037550151348114 
    
    TRAIN: epoch: 1200,  loss: 0.18318244814872742
    VAL: epoch: 1200,  val loss: 0.19298125803470612 
    
    TRAIN: epoch: 1225,  loss: 0.18787908554077148
    VAL: epoch: 1225,  val loss: 0.18469706177711487 
    
    TRAIN: epoch: 1250,  loss: 0.19416649639606476
    VAL: epoch: 1250,  val loss: 0.19308097660541534 
    
    TRAIN: epoch: 1275,  loss: 0.1898847371339798
    VAL: epoch: 1275,  val loss: 0.1908666044473648 
    
    TRAIN: epoch: 1300,  loss: 0.19189752638339996
    VAL: epoch: 1300,  val loss: 0.1874968707561493 
    
    TRAIN: epoch: 1325,  loss: 0.18831680715084076
    VAL: epoch: 1325,  val loss: 0.19845758378505707 
    
    TRAIN: epoch: 1350,  loss: 0.17979566752910614
    VAL: epoch: 1350,  val loss: 0.170597642660141 
    
    TRAIN: epoch: 1375,  loss: 0.20074152946472168
    VAL: epoch: 1375,  val loss: 0.18571996688842773 
    
    TRAIN: epoch: 1400,  loss: 0.1638781577348709
    VAL: epoch: 1400,  val loss: 0.16501015424728394 
    
    TRAIN: epoch: 1425,  loss: 0.18936039507389069
    VAL: epoch: 1425,  val loss: 0.17644105851650238 
    
    TRAIN: epoch: 1450,  loss: 0.20179620385169983
    VAL: epoch: 1450,  val loss: 0.18755610287189484 
    
    TRAIN: epoch: 1475,  loss: 0.1962258666753769
    VAL: epoch: 1475,  val loss: 0.1745985448360443 
    
    TRAIN: epoch: 1500,  loss: 0.16877909004688263
    VAL: epoch: 1500,  val loss: 0.15443237125873566 
    
    TRAIN: epoch: 1525,  loss: 0.19755364954471588
    VAL: epoch: 1525,  val loss: 0.195099338889122 
    
    TRAIN: epoch: 1550,  loss: 0.19540835916996002
    VAL: epoch: 1550,  val loss: 0.18495631217956543 
    
    TRAIN: epoch: 1575,  loss: 0.19159811735153198
    VAL: epoch: 1575,  val loss: 0.1827274113893509 
    
    TRAIN: epoch: 1600,  loss: 0.17775574326515198
    VAL: epoch: 1600,  val loss: 0.17288000881671906 
    
    TRAIN: epoch: 1625,  loss: 0.17260123789310455
    VAL: epoch: 1625,  val loss: 0.17612174153327942 
    
    TRAIN: epoch: 1650,  loss: 0.16365736722946167
    VAL: epoch: 1650,  val loss: 0.19712454080581665 
    
    TRAIN: epoch: 1675,  loss: 0.19552543759346008
    VAL: epoch: 1675,  val loss: 0.18845173716545105 
    
    TRAIN: epoch: 1700,  loss: 0.18566662073135376
    VAL: epoch: 1700,  val loss: 0.20166483521461487 
    
    TRAIN: epoch: 1725,  loss: 0.17784075438976288
    VAL: epoch: 1725,  val loss: 0.1740027219057083 
    
    TRAIN: epoch: 1750,  loss: 0.1627376675605774
    VAL: epoch: 1750,  val loss: 0.19868728518486023 
    
    TRAIN: epoch: 1775,  loss: 0.15076221525669098
    VAL: epoch: 1775,  val loss: 0.21826346218585968 
    
    TRAIN: epoch: 1800,  loss: 0.17026756703853607
    VAL: epoch: 1800,  val loss: 0.19661954045295715 
    
    TRAIN: epoch: 1825,  loss: 0.18085134029388428
    VAL: epoch: 1825,  val loss: 0.16413410007953644 
    
    TRAIN: epoch: 1850,  loss: 0.18280471861362457
    VAL: epoch: 1850,  val loss: 0.18638230860233307 
    
    TRAIN: epoch: 1875,  loss: 0.19526179134845734
    VAL: epoch: 1875,  val loss: 0.18105986714363098 
    
    TRAIN: epoch: 1900,  loss: 0.17651426792144775
    VAL: epoch: 1900,  val loss: 0.18618406355381012 
    
    TRAIN: epoch: 1925,  loss: 0.18326908349990845
    VAL: epoch: 1925,  val loss: 0.18428219854831696 
    
    TRAIN: epoch: 1950,  loss: 0.2005821168422699
    VAL: epoch: 1950,  val loss: 0.20368672907352448 
    
    TRAIN: epoch: 1975,  loss: 0.19552315771579742
    VAL: epoch: 1975,  val loss: 0.1984981745481491 
    
    TRAIN: epoch: 2000,  loss: 0.2751990854740143
    VAL: epoch: 2000,  val loss: 0.23844020068645477 
    
    TRAIN: epoch: 2025,  loss: 0.20003294944763184
    VAL: epoch: 2025,  val loss: 0.20878708362579346 
    
    TRAIN: epoch: 2050,  loss: 0.1995401233434677
    VAL: epoch: 2050,  val loss: 0.19300276041030884 
    
    TRAIN: epoch: 2075,  loss: 0.19434691965579987
    VAL: epoch: 2075,  val loss: 0.18510867655277252 
    
    TRAIN: epoch: 2100,  loss: 0.17462000250816345
    VAL: epoch: 2100,  val loss: 0.14868520200252533 
    
    TRAIN: epoch: 2125,  loss: 0.18490329384803772
    VAL: epoch: 2125,  val loss: 0.18901477754116058 
    
    TRAIN: epoch: 2150,  loss: 0.1579020768404007
    VAL: epoch: 2150,  val loss: 0.19376720488071442 
    
    TRAIN: epoch: 2175,  loss: 0.17812427878379822
    VAL: epoch: 2175,  val loss: 0.18545617163181305 
    
    TRAIN: epoch: 2200,  loss: 0.18837028741836548
    VAL: epoch: 2200,  val loss: 0.18206383287906647 
    
    TRAIN: epoch: 2225,  loss: 0.18815721571445465
    VAL: epoch: 2225,  val loss: 0.18832242488861084 
    
    TRAIN: epoch: 2250,  loss: 0.18343159556388855
    VAL: epoch: 2250,  val loss: 0.18309704959392548 
    
    TRAIN: epoch: 2275,  loss: 0.1864362359046936
    VAL: epoch: 2275,  val loss: 0.18803225457668304 
    
    TRAIN: epoch: 2300,  loss: 0.1997879594564438
    VAL: epoch: 2300,  val loss: 0.1937159299850464 
    
    TRAIN: epoch: 2325,  loss: 0.19716118276119232
    VAL: epoch: 2325,  val loss: 0.20744547247886658 
    
    TRAIN: epoch: 2350,  loss: 0.17482636868953705
    VAL: epoch: 2350,  val loss: 0.1944476217031479 
    
    TRAIN: epoch: 2375,  loss: 0.18908950686454773
    VAL: epoch: 2375,  val loss: 0.19335268437862396 
    
    TRAIN: epoch: 2400,  loss: 0.19176426529884338
    VAL: epoch: 2400,  val loss: 0.19198642671108246 
    
    TRAIN: epoch: 2425,  loss: 0.18702712655067444
    VAL: epoch: 2425,  val loss: 0.18612737953662872 
    
    TRAIN: epoch: 2450,  loss: 0.19161053001880646
    VAL: epoch: 2450,  val loss: 0.19319286942481995 
    
    TRAIN: epoch: 2475,  loss: 0.18313273787498474
    VAL: epoch: 2475,  val loss: 0.1856415867805481 
    
    TRAIN: epoch: 2500,  loss: 0.19300979375839233
    VAL: epoch: 2500,  val loss: 0.17183056473731995 
    
    TRAIN: epoch: 2525,  loss: 0.20024029910564423
    VAL: epoch: 2525,  val loss: 0.19651228189468384 
    
    TRAIN: epoch: 2550,  loss: 0.18575647473335266
    VAL: epoch: 2550,  val loss: 0.173621267080307 
    
    TRAIN: epoch: 2575,  loss: 0.17429104447364807
    VAL: epoch: 2575,  val loss: 0.17141291499137878 
    
    TRAIN: epoch: 2600,  loss: 0.17134925723075867
    VAL: epoch: 2600,  val loss: 0.1878400295972824 
    
    TRAIN: epoch: 2625,  loss: 0.12493643909692764
    VAL: epoch: 2625,  val loss: 0.1929841786623001 
    
    TRAIN: epoch: 2650,  loss: 0.1851094365119934
    VAL: epoch: 2650,  val loss: 0.16406822204589844 
    
    TRAIN: epoch: 2675,  loss: 0.18097920715808868
    VAL: epoch: 2675,  val loss: 0.1713145524263382 
    
    TRAIN: epoch: 2700,  loss: 0.1519523561000824
    VAL: epoch: 2700,  val loss: 0.15563663840293884 
    
    TRAIN: epoch: 2725,  loss: 0.18324986100196838
    VAL: epoch: 2725,  val loss: 0.1896202564239502 
    
    TRAIN: epoch: 2750,  loss: 0.14605537056922913
    VAL: epoch: 2750,  val loss: 0.18313410878181458 
    
    TRAIN: epoch: 2775,  loss: 0.11348693072795868
    VAL: epoch: 2775,  val loss: 0.1350107192993164 
    
    TRAIN: epoch: 2800,  loss: 0.15381193161010742
    VAL: epoch: 2800,  val loss: 0.12426949292421341 
    
    TRAIN: epoch: 2825,  loss: 0.18809448182582855
    VAL: epoch: 2825,  val loss: 0.18805494904518127 
    
    TRAIN: epoch: 2850,  loss: 0.11027433723211288
    VAL: epoch: 2850,  val loss: 0.07391321659088135 
    
    TRAIN: epoch: 2875,  loss: 0.05029531940817833
    VAL: epoch: 2875,  val loss: 0.047101691365242004 
    
    TRAIN: epoch: 2900,  loss: 0.194034606218338
    VAL: epoch: 2900,  val loss: 0.18472467362880707 
    
    TRAIN: epoch: 2925,  loss: 0.16798615455627441
    VAL: epoch: 2925,  val loss: 0.1640598326921463 
    
    TRAIN: epoch: 2950,  loss: 0.10300175100564957
    VAL: epoch: 2950,  val loss: 0.11353401839733124 
    
    TRAIN: epoch: 2975,  loss: 0.1880655735731125
    VAL: epoch: 2975,  val loss: 0.19832146167755127 
    
    TRAIN: epoch: 3000,  loss: 0.17675428092479706
    VAL: epoch: 3000,  val loss: 0.164883092045784 



```python
logger.finish()
```


<style>
    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}
    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }
    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }
    </style>
<div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>train loss</td><td>█▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁</td></tr><tr><td>val loss</td><td>█▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>train loss</td><td>0.17675</td></tr><tr><td>val loss</td><td>0.16488</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">small-lstm</strong> at: <a href='https://wandb.ai/shape-vs-texture/time-series/runs/tju1rve5/workspace' target="_blank">https://wandb.ai/shape-vs-texture/time-series/runs/tju1rve5/workspace</a><br/>Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20240402_102819-tju1rve5/logs</code>



```python
# model.load_state_dict(torch.load("checkpoints/0a4eos6o/500_epochs_0a4eos6o"))
```

Let's see how our model performs.

First of all we need to remember that if we feed a sequence to an LSTM model, then the prediction at point `n+1` will be based on all points before that. So when plotting the prediction the LSTM can't really mess up that much since at each step we predict on all the previous actual ground truth values.

No surprise then that the plot looks pretty good.


```python
t, signal = generate_signal(num_samples=SIGNAL_SIZE)
pred = model(signal.view(1, len(signal), 1))
predicted_signal = pred.view(len(signal)).detach()
predicted_signal = torch.roll(predicted_signal, 1, dims=0)

# Plotting
plt.figure(figsize=(15, 3))

plt.plot(t, signal, label='input')
plt.plot(t, predicted_signal, label='lstm predicted')
plt.legend(loc='best')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')


plt.tight_layout()
plt.show()
```


    
![png](assets/readme/output_33_0.png)
    


The real test is autoregressive generation, where we give the LSTM a part of the ground truth, and then use the LSTMs own output for step `n` to also predict `n+1` etc.

Let's see how we do in that case.


```python
def lstm_autoregressive_pred(signal, N):
    signal_continuation_autoreg = torch.clone(signal)
    signal_continuation_autoreg[N:] = 0
    for i in range(len(signal[N:])):
        pred = model(signal_continuation_autoreg.view(1, len(signal_continuation_autoreg), 1))
        pred = pred.view(len(signal_continuation_autoreg)).detach()
        signal_continuation_autoreg[N+i] = pred[N+i-1]
    return signal_continuation_autoreg

N=LOOKBACK_WINDOW_SIZE
for _ in range(10):
    t, signal = generate_signal(num_samples=SIGNAL_SIZE)
    plt.figure(figsize=(15, 3))

    plt.plot(t, signal, label='input')
    plt.plot(t, reconstruct_signal_fft(signal, N),'-x', label='fft reconstruction')
    plt.plot(t, lstm_autoregressive_pred(signal,N),'-x', label='lstm predicted')
    plt.legend(loc='best')
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')

    plt.axvspan(0, N-1, color='grey', alpha=0.3)


plt.tight_layout()
plt.show()
```


    
![png](assets/readme/output_35_0.png)
    



    
![png](assets/readme/output_35_1.png)
    



    
![png](assets/readme/output_35_2.png)
    



    
![png](assets/readme/output_35_3.png)
    



    
![png](assets/readme/output_35_4.png)
    



    
![png](assets/readme/output_35_5.png)
    



    
![png](assets/readme/output_35_6.png)
    



    
![png](assets/readme/output_35_7.png)
    



    
![png](assets/readme/output_35_8.png)
    



    
![png](assets/readme/output_35_9.png)
    


As one can see this is not exactly doing a good job.

### Sequence to Sequence

Perhaps the problem is that we are trying to use the model autoregressively to predict a longer sequence, even though it was trained to only ever make a prediction for the next elemen, so errors accumulate too much.

Let's try a different model architecture, one where we actually train it to produce a longer sequence directly.
Do to dis we'll use an LSTM to encode the lookback window into a hidden state and we'll use another LSTM to decode that into the predicted sequence.


```python
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
        decoder_input = signal[self.split_idx-1:]
        decoder_targets = torch.roll(decoder_input, -1, dims=0)
        if self.return_decoder_input:
            return torch.concat((encoder_input, decoder_input[:-1]), dim=0).unsqueeze_(dim=-1), decoder_targets[:-1].unsqueeze_(dim=-1)
        else:
            return encoder_input.unsqueeze_(dim=-1), decoder_targets[:-1].unsqueeze_(dim=-1)
```

Let's switch from LSTMs to GRUs as they are more parameter efficient.


```python
class GruEncoder(torch.nn.Module):
    def __init__(self, hidden_size) -> None:
        super(GruEncoder, self).__init__()
        self.gru = torch.nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True, num_layers=1)
        
    def forward(self, encoder_input):
        _, h_n = self.gru(encoder_input) # h_n =[1, N, h_dim]
        fft_result = torch.fft.rfft(encoder_input, dim=1, norm="forward")
        amplitudes = torch.abs(fft_result) # = [N, f_dim, 1]
        phases =  torch.angle(fft_result)
        return torch.concat((h_n, amplitudes.view(1,encoder_input.shape[0],-1), phases.view(1,encoder_input.shape[0],-1)), dim=-1).squeeze_(dim=0)
    
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
        self.enocoder_output_dim = self.encoder_hidden_dim + 2*(int(self.encoder_in_dim/2) + 1) # hidden dim + cat 2 * rfft

        self.encoder = GruEncoder(hidden_size=self.encoder_hidden_dim)
        self.decoder_cell = GruDecoderCell(hidden_size=self.enocoder_output_dim)
        self.leaky_relu = torch.nn.LeakyReLU()
        self.linear_1 = torch.nn.Linear(self.enocoder_output_dim, int(self.enocoder_output_dim/2))
        self.linear_2 = torch.nn.Linear(int(self.enocoder_output_dim/2), 1) 
        
    def forward(self, input): 
        if input.shape[1] == self.encoder_in_dim:
            encoder_input = input
            decoder_input = None
        elif input.shape[1] == self.encoder_in_dim + self.decoder_in_dim:
            encoder_input, decoder_input = input[:,:self.encoder_in_dim,:], input[:,self.encoder_in_dim:,:]
        else:
            raise "input shape mismatch"
        # assert encoder_input.shape[1] == self.encoder_in_dim
        # assert decoder_input.shape[1] == self.decoder_in_dim

        # Compress all of lookback window into hidden state with encoder and concat with fft
        h_n_augemented = self.encoder(encoder_input) # [N, h_encoder + fft]
        
        # init hidden state for decoder is the compressed lookback window hidden state from the encoder   
        h_decoder = h_n_augemented
        decoder_outputs = torch.empty((encoder_input.shape[0], self.decoder_in_dim, h_n_augemented.shape[-1]), dtype=torch.float) # [N, dec_in, h_enc+fft] 
        mlp_outputs = torch.empty((encoder_input.shape[0], self.decoder_in_dim, 1), dtype=torch.float) # [N, dec_in, 1] 
        # teacher forcing: feed actual sequence to decoder
        if decoder_input is not None: # Loop over inputs and feed them to decoder cell while updating the hidden state
            for i in range(self.decoder_in_dim):
                input_element = decoder_input[:,i,:]
                h_decoder = self.decoder_cell(input_element, h_decoder) # [N, h_enc + fft]
                decoder_outputs[:, i, :] = h_decoder
            x = self.linear_1(decoder_outputs)
            x = self.leaky_relu(x)
            x = self.linear_2(x)
            return x
        else: # No teacher forcing: autoregress
            # init the autoregression with the last value in the lookback window
            decoder_input = encoder_input[:,-1,:] #
            for i in range(self.decoder_in_dim):
                h_decoder = self.decoder_cell(decoder_input, h_decoder)
                # decoder_outputs[:, i, :] = h_decoder
                x = self.linear_1(h_decoder)
                x = self.leaky_relu(x)
                x = self.linear_2(x)
                decoder_input = x.clone().detach()
                mlp_outputs[:, i, :] = x
            return mlp_outputs
        
count_parameters(SeqToSeqGru(encoder_input_length=100, decoder_input_length=50))

```

    +---------------------------------+------------+
    |             Modules             | Parameters |
    +---------------------------------+------------+
    |     encoder.gru.weight_ih_l0    |     30     |
    |     encoder.gru.weight_hh_l0    |    300     |
    |      encoder.gru.bias_ih_l0     |     30     |
    |      encoder.gru.bias_hh_l0     |     30     |
    | decoder_cell.gru_cell.weight_ih |    336     |
    | decoder_cell.gru_cell.weight_hh |   37632    |
    |  decoder_cell.gru_cell.bias_ih  |    336     |
    |  decoder_cell.gru_cell.bias_hh  |    336     |
    |         linear_1.weight         |    6272    |
    |          linear_1.bias          |     56     |
    |         linear_2.weight         |     56     |
    |          linear_2.bias          |     1      |
    +---------------------------------+------------+
    Total Trainable Params: 45415





    45415




```python
LR = 0.002
SIGNAL_SIZE=105
LOOKBACK_WINDOW_SIZE=100
PREDICTION_SIZE=5
assert LOOKBACK_WINDOW_SIZE + PREDICTION_SIZE == SIGNAL_SIZE
EPOCH_FROM = 0
EPOCH_TO = 3000
SEND_TO_WANDB = True

#### BEGIN: Load model and init Logger
model = SeqToSeqGru(encoder_input_length=LOOKBACK_WINDOW_SIZE, decoder_input_length=PREDICTION_SIZE)

# checkpoint_path = './checkpoints/VVLZNW/1/500_epochs'
checkpoint_path = None

hyperparameters = [f"LR={LR}", f"PARAMS={count_parameters(model)}", f"SIGNAL_SIZE={SIGNAL_SIZE}", f"LOOKBACK_WINDOW_SIZE={LOOKBACK_WINDOW_SIZE}", f"PREDICTION_SIZE={PREDICTION_SIZE}"]

if checkpoint_path:
    EPOCH_FROM = int(checkpoint_path.split("/")[-1].split("_")[0])
    run_id = checkpoint_path.split("/")[-3]
    model.load_state_dict(torch.load(checkpoint_path))
    logger = Logger("enocder-decoder-lstm", send_to_wandb=SEND_TO_WANDB, id_resume=run_id, hyperparameters=hyperparameters)
else:
    logger = Logger("enocder-decoder-lstm", send_to_wandb=SEND_TO_WANDB, hyperparameters=hyperparameters)
### END
    
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
train_dataset = SeqToSeqDataset(size=1000, num_samples=SIGNAL_SIZE, split_idx=LOOKBACK_WINDOW_SIZE, return_decoder_input=True) 
eval_dataset = SeqToSeqDataset(size=1000, num_samples=SIGNAL_SIZE, split_idx=LOOKBACK_WINDOW_SIZE, return_decoder_input=False)  

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


```

    +---------------------------------+------------+
    |             Modules             | Parameters |
    +---------------------------------+------------+
    |     encoder.gru.weight_ih_l0    |     30     |
    |     encoder.gru.weight_hh_l0    |    300     |
    |      encoder.gru.bias_ih_l0     |     30     |
    |      encoder.gru.bias_hh_l0     |     30     |
    | decoder_cell.gru_cell.weight_ih |    336     |
    | decoder_cell.gru_cell.weight_hh |   37632    |
    |  decoder_cell.gru_cell.bias_ih  |    336     |
    |  decoder_cell.gru_cell.bias_hh  |    336     |
    |         linear_1.weight         |    6272    |
    |          linear_1.bias          |     56     |
    |         linear_2.weight         |     56     |
    |          linear_2.bias          |     1      |
    +---------------------------------+------------+
    Total Trainable Params: 45415



Tracking run with wandb version 0.16.5



Run data is saved locally in <code>/Users/roncaglionidaniele/Documents/projects/time-series-prediciton/wandb/run-20240402_110920-68ld8685</code>



Syncing run <strong><a href='https://wandb.ai/shape-vs-texture/time-series/runs/68ld8685/workspace' target="_blank">enocder-decoder-lstm</a></strong> to <a href='https://wandb.ai/shape-vs-texture/time-series' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/run' target="_blank">docs</a>)<br/>



View project at <a href='https://wandb.ai/shape-vs-texture/time-series' target="_blank">https://wandb.ai/shape-vs-texture/time-series</a>



View run at <a href='https://wandb.ai/shape-vs-texture/time-series/runs/68ld8685/workspace' target="_blank">https://wandb.ai/shape-vs-texture/time-series/runs/68ld8685/workspace</a>


    RUN ID CKOBK3
    Training from 0 to 3000
    DEVICE cpu
    TRAIN: epoch: 0,  loss: 4.388464450836182
    VAL: epoch: 0,  val loss: 4.319097518920898 
    
    TRAIN: epoch: 25,  loss: 0.5327506065368652
    VAL: epoch: 25,  val loss: 2.443162679672241 
    
    TRAIN: epoch: 50,  loss: 0.2513550817966461
    VAL: epoch: 50,  val loss: 2.1959521770477295 
    
    TRAIN: epoch: 75,  loss: 0.13654153048992157
    VAL: epoch: 75,  val loss: 2.72782039642334 
    
    TRAIN: epoch: 100,  loss: 0.09969203174114227
    VAL: epoch: 100,  val loss: 1.4660788774490356 
    
    TRAIN: epoch: 125,  loss: 0.07858271896839142
    VAL: epoch: 125,  val loss: 1.5324313640594482 
    
    TRAIN: epoch: 150,  loss: 0.08868931233882904
    VAL: epoch: 150,  val loss: 1.6074926853179932 
    
    TRAIN: epoch: 175,  loss: 0.06154822185635567
    VAL: epoch: 175,  val loss: 1.3860915899276733 
    
    TRAIN: epoch: 200,  loss: 0.09157481789588928
    VAL: epoch: 200,  val loss: 1.243523359298706 
    
    TRAIN: epoch: 225,  loss: 0.06291664391756058
    VAL: epoch: 225,  val loss: 1.3807008266448975 
    
    TRAIN: epoch: 250,  loss: 0.07391904294490814
    VAL: epoch: 250,  val loss: 1.2689239978790283 
    
    TRAIN: epoch: 275,  loss: 0.060623377561569214
    VAL: epoch: 275,  val loss: 1.6078095436096191 
    
    TRAIN: epoch: 300,  loss: 0.04606320708990097
    VAL: epoch: 300,  val loss: 1.11183762550354 
    
    TRAIN: epoch: 325,  loss: 0.046013105660676956
    VAL: epoch: 325,  val loss: 0.9016784429550171 
    
    TRAIN: epoch: 350,  loss: 0.04353608563542366
    VAL: epoch: 350,  val loss: 1.0384902954101562 
    
    TRAIN: epoch: 375,  loss: 0.03476811200380325
    VAL: epoch: 375,  val loss: 0.9717272520065308 
    
    TRAIN: epoch: 400,  loss: 0.0411275178194046
    VAL: epoch: 400,  val loss: 0.9282732605934143 
    
    TRAIN: epoch: 425,  loss: 0.0382181853055954
    VAL: epoch: 425,  val loss: 1.943680763244629 
    
    TRAIN: epoch: 450,  loss: 0.0397186279296875
    VAL: epoch: 450,  val loss: 1.251766562461853 
    
    TRAIN: epoch: 475,  loss: 0.04231695458292961
    VAL: epoch: 475,  val loss: 0.9488347768783569 
    
    TRAIN: epoch: 500,  loss: 0.038923121988773346
    VAL: epoch: 500,  val loss: 1.098284363746643 
    
    TRAIN: epoch: 525,  loss: 0.03671984747052193
    VAL: epoch: 525,  val loss: 0.8273300528526306 
    
    TRAIN: epoch: 550,  loss: 0.027758119627833366
    VAL: epoch: 550,  val loss: 0.8310756087303162 
    
    TRAIN: epoch: 575,  loss: 0.0334911122918129
    VAL: epoch: 575,  val loss: 1.0694448947906494 
    
    TRAIN: epoch: 600,  loss: 0.031527046114206314
    VAL: epoch: 600,  val loss: 0.8221850991249084 
    
    TRAIN: epoch: 625,  loss: 0.031596794724464417
    VAL: epoch: 625,  val loss: 0.927087664604187 
    
    TRAIN: epoch: 650,  loss: 0.03248303011059761
    VAL: epoch: 650,  val loss: 0.8324424624443054 
    
    TRAIN: epoch: 675,  loss: 0.03410637378692627
    VAL: epoch: 675,  val loss: 0.827475905418396 
    
    TRAIN: epoch: 700,  loss: 0.02646588906645775
    VAL: epoch: 700,  val loss: 0.6460319757461548 
    
    TRAIN: epoch: 725,  loss: 0.02657145820558071
    VAL: epoch: 725,  val loss: 0.7529770731925964 
    
    TRAIN: epoch: 750,  loss: 0.0301309023052454
    VAL: epoch: 750,  val loss: 0.6842729449272156 
    
    TRAIN: epoch: 775,  loss: 0.037611667066812515
    VAL: epoch: 775,  val loss: 1.0095769166946411 
    
    TRAIN: epoch: 800,  loss: 0.023726170882582664
    VAL: epoch: 800,  val loss: 0.8534400463104248 
    
    TRAIN: epoch: 825,  loss: 0.02734798938035965
    VAL: epoch: 825,  val loss: 0.741452693939209 
    
    TRAIN: epoch: 850,  loss: 0.031023958697915077
    VAL: epoch: 850,  val loss: 0.9470599293708801 
    
    TRAIN: epoch: 875,  loss: 0.027525166049599648
    VAL: epoch: 875,  val loss: 0.9846819639205933 
    
    TRAIN: epoch: 900,  loss: 0.024070600047707558
    VAL: epoch: 900,  val loss: 0.6860682368278503 
    
    TRAIN: epoch: 925,  loss: 0.02625911869108677
    VAL: epoch: 925,  val loss: 1.4589616060256958 
    
    TRAIN: epoch: 950,  loss: 0.022479204460978508
    VAL: epoch: 950,  val loss: 0.7840378284454346 
    
    TRAIN: epoch: 975,  loss: 0.028533736243844032
    VAL: epoch: 975,  val loss: 0.7740046381950378 
    
    TRAIN: epoch: 1000,  loss: 0.02955360896885395
    VAL: epoch: 1000,  val loss: 0.7720658183097839 
    
    TRAIN: epoch: 1025,  loss: 0.02104916051030159
    VAL: epoch: 1025,  val loss: 0.5890712738037109 
    
    TRAIN: epoch: 1050,  loss: 0.026371566578745842
    VAL: epoch: 1050,  val loss: 1.150357961654663 
    
    TRAIN: epoch: 1075,  loss: 0.01658649556338787
    VAL: epoch: 1075,  val loss: 0.7524160146713257 
    
    TRAIN: epoch: 1100,  loss: 0.022653736174106598
    VAL: epoch: 1100,  val loss: 0.825019896030426 
    
    TRAIN: epoch: 1125,  loss: 0.02065170556306839
    VAL: epoch: 1125,  val loss: 0.8212536573410034 
    
    TRAIN: epoch: 1150,  loss: 0.024800190702080727
    VAL: epoch: 1150,  val loss: 0.6237711906433105 
    
    TRAIN: epoch: 1175,  loss: 0.019234512001276016
    VAL: epoch: 1175,  val loss: 0.9285646677017212 
    
    TRAIN: epoch: 1200,  loss: 0.029563700780272484
    VAL: epoch: 1200,  val loss: 1.0221290588378906 
    
    TRAIN: epoch: 1225,  loss: 0.019891254603862762
    VAL: epoch: 1225,  val loss: 0.8699361681938171 
    
    TRAIN: epoch: 1250,  loss: 0.023196285590529442
    VAL: epoch: 1250,  val loss: 0.8114435076713562 
    
    TRAIN: epoch: 1275,  loss: 0.023485802114009857
    VAL: epoch: 1275,  val loss: 0.8576446771621704 
    
    TRAIN: epoch: 1300,  loss: 0.027229387313127518
    VAL: epoch: 1300,  val loss: 1.0048959255218506 
    
    TRAIN: epoch: 1325,  loss: 0.01693608984351158
    VAL: epoch: 1325,  val loss: 1.0074968338012695 
    
    TRAIN: epoch: 1350,  loss: 0.027260087430477142
    VAL: epoch: 1350,  val loss: 0.7010161280632019 
    
    TRAIN: epoch: 1375,  loss: 0.024237973615527153
    VAL: epoch: 1375,  val loss: 1.7908803224563599 
    
    TRAIN: epoch: 1400,  loss: 0.019285550341010094
    VAL: epoch: 1400,  val loss: 1.1268352270126343 
    
    TRAIN: epoch: 1425,  loss: 0.017980819568037987
    VAL: epoch: 1425,  val loss: 0.5795556902885437 
    
    TRAIN: epoch: 1450,  loss: 0.0166578721255064
    VAL: epoch: 1450,  val loss: 0.7341338992118835 
    
    TRAIN: epoch: 1475,  loss: 0.023523306474089622
    VAL: epoch: 1475,  val loss: 0.6681957244873047 
    
    TRAIN: epoch: 1500,  loss: 0.019787132740020752
    VAL: epoch: 1500,  val loss: 1.1680368185043335 
    
    TRAIN: epoch: 1525,  loss: 0.025078674778342247
    VAL: epoch: 1525,  val loss: 1.004294753074646 
    
    TRAIN: epoch: 1550,  loss: 0.026554284617304802
    VAL: epoch: 1550,  val loss: 0.7054751515388489 
    
    TRAIN: epoch: 1575,  loss: 0.02069822885096073
    VAL: epoch: 1575,  val loss: 1.9088228940963745 
    
    TRAIN: epoch: 1600,  loss: 0.020546577870845795
    VAL: epoch: 1600,  val loss: 0.7405506372451782 
    
    TRAIN: epoch: 1625,  loss: 0.01665620505809784
    VAL: epoch: 1625,  val loss: 1.2045737504959106 
    
    TRAIN: epoch: 1650,  loss: 0.02229423075914383
    VAL: epoch: 1650,  val loss: 0.8671076893806458 
    
    TRAIN: epoch: 1675,  loss: 0.02262669987976551
    VAL: epoch: 1675,  val loss: 0.6581881046295166 
    
    TRAIN: epoch: 1700,  loss: 0.022058676928281784
    VAL: epoch: 1700,  val loss: 0.7357039451599121 
    
    TRAIN: epoch: 1725,  loss: 0.01813415251672268
    VAL: epoch: 1725,  val loss: 0.5730293393135071 
    
    TRAIN: epoch: 1750,  loss: 0.015937278047204018
    VAL: epoch: 1750,  val loss: 0.6831682324409485 
    
    TRAIN: epoch: 1775,  loss: 0.022450122982263565
    VAL: epoch: 1775,  val loss: 0.8145224452018738 
    
    TRAIN: epoch: 1800,  loss: 0.018176790326833725
    VAL: epoch: 1800,  val loss: 0.7258558869361877 
    
    TRAIN: epoch: 1825,  loss: 0.01798907481133938
    VAL: epoch: 1825,  val loss: 0.7879306077957153 
    
    TRAIN: epoch: 1850,  loss: 0.018691357225179672
    VAL: epoch: 1850,  val loss: 0.6294786334037781 
    
    TRAIN: epoch: 1875,  loss: 0.014679720625281334
    VAL: epoch: 1875,  val loss: 0.5741575360298157 
    
    TRAIN: epoch: 1900,  loss: 0.01691582053899765
    VAL: epoch: 1900,  val loss: 0.7364760637283325 
    
    TRAIN: epoch: 1925,  loss: 0.018724389374256134
    VAL: epoch: 1925,  val loss: 0.7957779169082642 
    
    TRAIN: epoch: 1950,  loss: 0.019083570688962936
    VAL: epoch: 1950,  val loss: 0.997842013835907 
    
    TRAIN: epoch: 1975,  loss: 0.01892121508717537
    VAL: epoch: 1975,  val loss: 0.8091790080070496 
    
    TRAIN: epoch: 2000,  loss: 0.017458854243159294
    VAL: epoch: 2000,  val loss: 0.8658117651939392 
    
    TRAIN: epoch: 2025,  loss: 0.015591239556670189
    VAL: epoch: 2025,  val loss: 0.7734180688858032 
    
    TRAIN: epoch: 2050,  loss: 0.017117789015173912
    VAL: epoch: 2050,  val loss: 0.7078442573547363 
    
    TRAIN: epoch: 2075,  loss: 0.01730738952755928
    VAL: epoch: 2075,  val loss: 0.6624214053153992 
    
    TRAIN: epoch: 2100,  loss: 0.018388979136943817
    VAL: epoch: 2100,  val loss: 0.7004076242446899 
    
    TRAIN: epoch: 2125,  loss: 0.018360784277319908
    VAL: epoch: 2125,  val loss: 0.5784444212913513 
    
    TRAIN: epoch: 2150,  loss: 0.017133114859461784
    VAL: epoch: 2150,  val loss: 0.6977314949035645 
    
    TRAIN: epoch: 2175,  loss: 0.014634333550930023
    VAL: epoch: 2175,  val loss: 0.5678344368934631 
    
    TRAIN: epoch: 2200,  loss: 0.01429123617708683
    VAL: epoch: 2200,  val loss: 0.7410897016525269 
    
    TRAIN: epoch: 2225,  loss: 0.018269270658493042
    VAL: epoch: 2225,  val loss: 0.763007640838623 
    
    TRAIN: epoch: 2250,  loss: 0.014326775446534157
    VAL: epoch: 2250,  val loss: 0.6033219695091248 
    
    TRAIN: epoch: 2275,  loss: 0.013659998774528503
    VAL: epoch: 2275,  val loss: 0.5824646353721619 
    
    TRAIN: epoch: 2300,  loss: 0.026099424809217453
    VAL: epoch: 2300,  val loss: 0.9253022074699402 
    
    TRAIN: epoch: 2325,  loss: 0.016598496586084366
    VAL: epoch: 2325,  val loss: 0.6191789507865906 
    
    TRAIN: epoch: 2350,  loss: 0.014597075991332531
    VAL: epoch: 2350,  val loss: 0.5836796760559082 
    
    TRAIN: epoch: 2375,  loss: 0.021042125299572945
    VAL: epoch: 2375,  val loss: 0.8644558787345886 
    
    TRAIN: epoch: 2400,  loss: 0.011771935038268566
    VAL: epoch: 2400,  val loss: 0.5839709043502808 
    
    TRAIN: epoch: 2425,  loss: 0.012463554739952087
    VAL: epoch: 2425,  val loss: 0.6523046493530273 
    
    TRAIN: epoch: 2450,  loss: 0.013305990025401115
    VAL: epoch: 2450,  val loss: 0.775132954120636 
    
    TRAIN: epoch: 2475,  loss: 0.016182081773877144
    VAL: epoch: 2475,  val loss: 1.3218971490859985 
    
    TRAIN: epoch: 2500,  loss: 0.012872408144176006
    VAL: epoch: 2500,  val loss: 0.5861882567405701 
    
    TRAIN: epoch: 2525,  loss: 0.01611347496509552
    VAL: epoch: 2525,  val loss: 0.6087290644645691 
    
    TRAIN: epoch: 2550,  loss: 0.013257384300231934
    VAL: epoch: 2550,  val loss: 0.5230178833007812 
    
    TRAIN: epoch: 2575,  loss: 0.014425327070057392
    VAL: epoch: 2575,  val loss: 0.6023425459861755 
    
    TRAIN: epoch: 2600,  loss: 0.01189226470887661
    VAL: epoch: 2600,  val loss: 0.5269597172737122 
    
    TRAIN: epoch: 2625,  loss: 0.013735225424170494
    VAL: epoch: 2625,  val loss: 0.487011194229126 
    
    TRAIN: epoch: 2650,  loss: 0.013472639955580235
    VAL: epoch: 2650,  val loss: 0.7257061004638672 
    
    TRAIN: epoch: 2675,  loss: 0.016370704397559166
    VAL: epoch: 2675,  val loss: 0.5091091394424438 
    
    TRAIN: epoch: 2700,  loss: 0.01125921681523323
    VAL: epoch: 2700,  val loss: 0.536699116230011 
    
    TRAIN: epoch: 2725,  loss: 0.01994224078953266
    VAL: epoch: 2725,  val loss: 1.1379815340042114 
    
    TRAIN: epoch: 2750,  loss: 0.01313815452158451
    VAL: epoch: 2750,  val loss: 1.156205177307129 
    
    TRAIN: epoch: 2775,  loss: 0.013818405568599701
    VAL: epoch: 2775,  val loss: 0.5278841257095337 
    
    TRAIN: epoch: 2800,  loss: 0.018642893061041832
    VAL: epoch: 2800,  val loss: 1.1123050451278687 
    
    TRAIN: epoch: 2825,  loss: 0.014360545203089714
    VAL: epoch: 2825,  val loss: 0.8821943998336792 
    
    TRAIN: epoch: 2850,  loss: 0.013552120886743069
    VAL: epoch: 2850,  val loss: 0.7062250375747681 
    
    TRAIN: epoch: 2875,  loss: 0.013273481279611588
    VAL: epoch: 2875,  val loss: 0.4934003949165344 
    
    TRAIN: epoch: 2900,  loss: 0.01654778979718685
    VAL: epoch: 2900,  val loss: 0.4627736806869507 
    
    TRAIN: epoch: 2925,  loss: 0.012376263737678528
    VAL: epoch: 2925,  val loss: 0.4696871042251587 
    
    TRAIN: epoch: 2950,  loss: 0.019997209310531616
    VAL: epoch: 2950,  val loss: 0.576504111289978 
    
    TRAIN: epoch: 2975,  loss: 0.013129068538546562
    VAL: epoch: 2975,  val loss: 0.8681044578552246 
    
    TRAIN: epoch: 3000,  loss: 0.01448422484099865
    VAL: epoch: 3000,  val loss: 0.5304955840110779 



```python
logger.finish()
```


<style>
    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}
    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }
    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }
    </style>
<div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>train loss</td><td>█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁</td></tr><tr><td>val loss</td><td>█▅▃▃▂▂▂▂▂▂▁▂▁▂▂▁▂▂▃▁▂▂▁▁▂▁▂▁▁▂▂▂▂▁▁▁▂▂▁▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>train loss</td><td>0.01448</td></tr><tr><td>val loss</td><td>0.5305</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">enocder-decoder-lstm</strong> at: <a href='https://wandb.ai/shape-vs-texture/time-series/runs/68ld8685/workspace' target="_blank">https://wandb.ai/shape-vs-texture/time-series/runs/68ld8685/workspace</a><br/>Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20240402_110920-68ld8685/logs</code>



```python
def lstm_pred(signal, N):
    signal = torch.clone(signal)
    pred = model(signal[:N].view(1,N,1))
    return pred.view(SIGNAL_SIZE-N).detach()

for _ in range(10):
    t, signal = generate_signal(num_samples=SIGNAL_SIZE, periods_range=(2,100))
    plt.figure(figsize=(15, 3))

    plt.plot(t, signal, label='input')
    plt.plot(t, reconstruct_signal_fft(signal, LOOKBACK_WINDOW_SIZE),'-x', label='fft reconstruction')
    plt.plot(range(LOOKBACK_WINDOW_SIZE, SIGNAL_SIZE), lstm_pred(signal, LOOKBACK_WINDOW_SIZE),'-x', label='lstm predicted')
    plt.legend(loc='best')
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')

    plt.axvspan(0, LOOKBACK_WINDOW_SIZE-1, color='grey', alpha=0.3)


plt.tight_layout()
plt.show()
```


    
![png](assets/readme/output_43_0.png)
    



    
![png](assets/readme/output_43_1.png)
    



    
![png](assets/readme/output_43_2.png)
    



    
![png](assets/readme/output_43_3.png)
    



    
![png](assets/readme/output_43_4.png)
    



    
![png](assets/readme/output_43_5.png)
    



    
![png](assets/readme/output_43_6.png)
    



    
![png](assets/readme/output_43_7.png)
    



    
![png](assets/readme/output_43_8.png)
    



    
![png](assets/readme/output_43_9.png)
    


As we can see this already looks much better.

## Training a larger model on more complex signals

Above we have trained  small models of about 40k parameters for a very short time.
For my final experiment we:
- use GRUs rather than LSTMs as they are more parameter efficient
- do away with the GRU encoder and rather embed the lookback window + fft with a faster MLP
- do away with teacher forcing in the decoder during training
- train our model progressively until we have model that can handle up to 45 different components
- train on about 30 million synthetic samples with a lookback of 100 samples and predicting the next 10 samples

For this las experiment see `train.py`.

### Results

Preliminary results are satisfactory in that the model starts to predict the signal fairly accurately even for signals with up to 45 harmonic components. I did not train to full convergence, there was still room to improve the performance quite a bit I believe judging from the training behaviour and the validation curves, and the examples below show that the model either already nails the prediction perfectly or it clearly has the right intuition about where the signal will go but can't quite nail the values perfectly yet.

(*in legend lstm should actually be gru)

![test 1](assets/test1.png)
![test 2](assets/test2.png)




```python

```
