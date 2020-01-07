import scipy
import matplotlib.pyplot as plt
import scipy.io.wavfile

sample_rate, X = scipy.io.wavfile.read('595.wav')
print (sample_rate, X.shape )
plt.specgram(X, Fs=sample_rate, xextent=(0,30))