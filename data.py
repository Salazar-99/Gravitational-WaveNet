import numpy as np 
from pycbc.waveform import get_td_waveform

def generate_data(batch_size, sample_rate, mass_range):
    '''
    Function for generating synthetic gravitational waveforms
    for binary mergers of equal masses over a range of masses determined by batch_size and mass_range
    as well as positive class labels. The waveforms are clipped/padded to be 256 steps in length.
    Param: batch_size - Integer number of waveforms to produce
    Param: sample_rate (Hz) - Integer number of measurements per second
    Param: mass_range (solar masses) - List of form [lower_bound, upper_bound] for mass range
    Param: data - list containing generated waveforms as numpy arrays
    '''
    data = []
    mass_step = (mass_range[1]-mass_range[0])/batch_size
    masses = np.arange(mass_range[0], mass_range[1], mass_step)
    for mass in masses:
        #Generating waveform
        wave, _ = get_td_waveform(approximant='IMRPhenomD', mass1=mass, mass2=mass, delta_t=1.0/sample_rate, f_lower=25)
        #Cropping/padding to get waveforms of length 256
        if len(wave) > 256:
            wave = wave[len(wave)-256:]
        else:
            wave = np.concatenate([np.zeros(256-len(wave)), wave])
        #Normalizing waveform
        wave = wave/max(np.correlate(wave, wave, mode="full"))**0.5
        #Saving waveform as numpy array
        data.append(np.asarray(wave))
    labels = np.ones(len(data))
    return np.asarray(data), labels

def generate_noise(batch_size, sd=1):
    '''
    Function for generating sequences of white noise of fixed length (256) and negative class labels
    Param: batch_size - Integer number of noise sequences to produce
    Param: sd - Standard deviation of gaussian noise produced
    '''
    noise = []
    for i in range(batch_size):
        temp = np.random.normal(scale=sd, size=[256])
        noise.append(temp)
    labels = np.zeros(batch_size)
    return noise, labels

def generate_noisy_data(snr, batch_size, sample_rate, mass_range):
    '''
    Function for generating synthetic gravitational waveforms
    with additive gaussian noise of a specified signal-to-noise ratio, positive class labels,
    guassian noise with fixed power, and negative class labels
    Param: snr - Signal to noise ratio
    Params: The rest are identical to generate_data()
    '''
    #Generate pure waveforms and labels
    raw_data, data_labels = generate_data(batch_size, sample_rate, mass_range)
    #Compute average power of pure waveforms
    power = avg_power(raw_data)
    #Compute required noise power to achieve specified SNR
    sd = np.sqrt(snr*power)
    #Generating noise of specified power
    signal_noise, _ = generate_noise(batch_size, sd)
    #Combining the data and the noise
    noisy_data = raw_data + signal_noise
    return noisy_data

def avg_power(data):
    '''
    Function for computing the average power of an array of waveforms
    Param: data - Numpy array containing numpy arrays of waveforms
    '''
    powers = []
    #Calculate power of each waveform
    for wave in data:
        powers.append(np.sum(np.square(wave))/256)
    #Return mean of powers
    return np.mean(powers)

def save_data(data, sample_rate, mass_range, batch_size, snr):
    '''
    Function for saving generated data to a numpy file
    in data directory for reproducability
    Param: data - output of generate_data()
    Params: The rest are identical to generate_data()
    '''
    #Generating filename containing parameters used to create data
    file = f"data/data-snr:{snr}-sr:{sample_rate}-mr:({mass_range[0]}-{mass_range[1]})-bs:{batch_size}" 
    #Saving data
    np.save(file, data)

def combine_data(data, noise, data_labels, noise_labels):
    '''
    Function for combining data, noise, and labels into
    corresponding numpy arrays for training the model
    Param: data - List of waveforms
    Param: data_labels - Numpy array of 1's for positive class
    Param: noise - list of noise sequences
    Param: noise_labels - Numpy array of 0's for negative class
    Output: X - Numpy array of numpy arrays of data/noise
    Output: y - numpy array containing labels
    '''
    X = []
    #First waves then noise
    for wave in data:
        X.append(wave)
    for sequence in noise:
        X.append(sequence)
    y = []
    for label in data_labels:
        y.append(label)
    for label in noise_labels:
        y.append(label)
    return np.asarray(X)[..., np.newaxis], np.asarray(y)