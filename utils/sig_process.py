import numpy as np
from scipy.signal import butter	#, lfilter, periodogram

class lFilter:
	def __init__(self, lowcut, highcut, sample_rate, order=2):
		nyq = 0.5 * sample_rate
		low = lowcut / nyq
		high = highcut / nyq
		b, a = butter(order, [low, high], btype='band')
		self.coefA = a
		self.coefB = b
		self.order = len(self.coefA) - 1
		self.z = [0] * self.order

	def lfilt(self, data):
		y = (data * self.coefB[0]) + self.z[0]
		for i in range(0, self.order):
			if (i < self.order - 1):
				self.z[i] = (data * self.coefB[i+1]) + (self.z[i+1]) - (self.coefA[i+1] * y)
			else:
				self.z[i] = (data * self.coefB[i+1]) - (self.coefA[i+1] * y)
		return y