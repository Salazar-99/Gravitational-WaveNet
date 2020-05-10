from data import *
import argparse

#Argument parser 
parser = argparse.ArgumentParser(description='Specify model hyperparameters and data path')

#Model args
parser.add_argument('--snr', type=float, help="Specify the signal to noise ratio used to generate the data and noise waveforms")
parser.add_argument('--batch_size', type=int, help="Specify integer number of waveforms to create")
parser.add_argument('--sample_rate', type=int, help="Specify integer sample rate for generated waveforms")
parser.add_argument('--mass_range', nargs='+', type=int, help="List containing upper and lower bounds for masses of generated waveforms")

#Collect args
args = parser.parse_args()

#Generate the data
data = generate_noisy_data(args.snr, args.batch_size, args.sample_rate, args.mass_range)

#Save the data
save_data(data, args.sample_rate, args.mass_range, args.batch_size, args.snr)