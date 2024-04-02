import torch
from train_decoder_gru import SeqToSeqGru
from utils import plot_predictions

model = SeqToSeqGru(encoder_input_length=100, decoder_input_length=20)
model.load_state_dict(torch.load('./checkpoints/wqbw5aev/02-04-2024_16-21-27/500_epochs'))

plot_predictions(model, 120, 100)
