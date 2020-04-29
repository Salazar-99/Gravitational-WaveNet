import numpy as np 
from pycbc.waveform import get_td_waveform

def generate_data(sample_rate, mass_range, mass_step):
    '''
    Function for generating synthetic gravitational waveforms
    for binary mergers of equal masses over a specified range of masses.
    Param: sample_rate (Hz) - Integer number of measurements per second
    Param: mass_range (solar masses) - List of form [lower_bound, upper_bound] for mass range
    Param: mass_step (solar masses) - Float indicating step size for mass range
    Param: data - list containing generated waveforms as numpy arrays
    '''
    data = []
    masses = np.arange(mass_range[0], mass_range[1], mass_step)
    for mass in masses:
        #Generating waveform
        wave, _ = get_td_waveform(approximant='IMRPhenomD', mass1=mass, mass2=mass, delta_t=1.0/sample_rate, f_lower=25)
        #Normalizing waveform
        wave = wave/max(np.correlate(wave, wave, mode="full"))**0.5
        #Saving waveform as numpy array
        data.append(wave)
    return data

def generate_noise(sample_rate, length):
    '''
    Function for generating sequences of white noise
    Param: sample_rate (Hz) - Integer number of measurements per second
    Param: length (s) - Length of sequence in seconds
    '''
    noise = np.random.normal(size=[sample_rate*length])
    return noise

def save_data(data, sample_rate, mass_range, mass_step):
    '''
    Function for saving generated data to a numpy file
    in data directory for repeatability
    Param: data - output of generate_data()
    Params: The rest are identical to generate_data()
    '''
    #Generating filename containing parameters used to create data
    file = f"data/data-sr:{sample_rate}-mr:({mass_range[0]}-{mass_range[1]})-ms:{mass_step}" 
    #Saving data
    np.save(file, data)