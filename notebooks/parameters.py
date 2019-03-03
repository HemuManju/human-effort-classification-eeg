# Here are the parameters used in the analysis 
# This is a safe way to set parameters because same parameters are used in all the analysis

epoch_length = 1
image_size = 32
subjects = ['7707', '7708', '8801', '8802', '8803', '8815', '8819', '8820', '8821',
            '8822', '8823', '8824', '8826', '8828', '8830', '8831', '8832', '8833']
trials = ['HighFine', 'HighGross', 'LowFine', 'LowGross']
freq_bands = [[4, 7], [8, 10], [11, 13], [14, 22], [23, 35], [35, 45]]
n_freqs = len(freq_bands)  # number of frequency bands
n_electrodes = 20
s_freq = 256
