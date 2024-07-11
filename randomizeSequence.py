from matplotlib.pyplot import legend
from ScrappySequence import *
import random
import math
from typing import Union

def removeInitial6As(s: ScrappySequence):
    return ScrappySequence.fromSequenceItems(s.slice(6, len(s) - 6))


def randomizeSeparator(s: ScrappySequence, tolerance: int, merLen=12):
    assert 0 <= tolerance <= 8
    left = random.randrange(8 - tolerance, 8 + tolerance)
    right = random.randrange(8 + merLen - tolerance, 8 + merLen + tolerance)

    return ScrappySequence.fromSequenceItems(s.slice(left, right))


def randomizeSeparatorAsymmetric(
    s: ScrappySequence, startLeftTol, startRightTol, endLeftTol, endRightTol, merLen=12
):
    assert 0 <= startLeftTol <= 8
    assert 0 <= startRightTol <= 8
    assert 0 <= endLeftTol <= 8
    assert 0 <= endRightTol <= 8

    left = random.randrange(8 - startLeftTol, 8 + startRightTol)
    right = random.randrange(8 + merLen - endLeftTol, 8 + merLen + endRightTol)

    return ScrappySequence.fromSequenceItems(s.slice(left, right))


def randomizeSeparatorAsymmetricSignal(
    s: ScrappySequence, startLeftTol, startRightTol, endLeftTol, endRightTol, merLen=12
):
    randomizedBases = randomizeSeparatorAsymmetric(
        s, startLeftTol, startRightTol, endLeftTol, endRightTol, merLen
    )
    signal = randomizedBases.generateRandomSignal()

    # count how many samples were generated from the first letter
    firstIndexLen = 0
    while signal.indexes[firstIndexLen] == signal.indexes[0]:
        firstIndexLen += 1

    # count how many samples were generated from the last letter
    lastIndexLen = 0
    while (
        signal.indexes[len(signal.indexes) - 1 - lastIndexLen]
        == signal.indexes[len(signal.indexes) - 1]
    ):
        lastIndexLen += 1

    start = random.randrange(0, firstIndexLen)
    end = random.randrange(0, lastIndexLen)

    return signal.signal[start : len(signal.signal) - 1 - end]


def randomizeSequence(s: ScrappySequence):

    points = [0, 0]
    while points[0] == points[1]:
        points = [random.randrange(0, len(s)) for i in range(2)]
        points.sort()

    return ScrappySequence.fromSequenceItems(s.slice(points[0], points[1]))

import abc
class Range(metaclass=abc.ABCMeta):
	def __init__(self, start, end):
		self.start = start
		self.end = end

	@abc.abstractmethod
	def rand(self) -> 'Union[float, int]':
		return 
	
class IntRange(Range):
	def rand(self) -> int:
		return random.randrange(self.start, self.end)

class FloatRange(Range):
	def rand(self) -> float :
		return random.uniform(self.start, self.end)

# This class uses the builder pattern to create a 
# sequential pipe of functions that are applied to an input signa;
class SignalRandomizer:
	SUGGESTED_HEIGHT_THRESHOLD = 1.25

	def __init__(self) -> None:
		self.pipe = []
		self.built = False

	"""
	This function adds data points when raises or dips bigger than a threshold
	are detected. The data points are generated randomly between the two ends of the
	raise/drop. The amount of data points is controlled by samples_amount. 
	"""
	def add_samples_in_big_jumps(self, 
		threshold:float=SUGGESTED_HEIGHT_THRESHOLD,
		samples_amount:IntRange=IntRange(1, 4),
	 ) -> 'SignalRandomizer':

		def f(signal:'list[float]') -> 'list[float]':
			if len(signal) == 0:
				return signal

			new_sig = []
			
			for i in range(1, len(signal)):
				if abs(signal[i] - signal[i-1]) > threshold:
					valueGenerator = FloatRange(signal[i], signal[i-1])
					values = []
					for _ in range(samples_amount.rand()):
						values.append(valueGenerator.rand())
					
					values.sort()
					new_sig.extend(values)
				
				new_sig.append(signal[i])
				

			return new_sig

		self._add(f)
		return self

	"""
	This function shifts a signal up or down by a random amount between the given range.
	All data points are shifted equally.
	"""
	def shift_entire_signal(self, shift_range=FloatRange(-0.05, 0.05)) -> 'SignalRandomizer':
		def f(signal:'list[float]') -> 'list[float]':
			shift = shift_range.rand()
			return [i + shift for i in signal]

		self._add(f)
		return self

	"""
	This function shifts flat regions of a signal up or down by a random amount between the given range.
	Every flat region is shifted differently.
	"""
	def shift_flat_regions(self, 
		shift_range:Range=FloatRange(-0.1, 0.1), 
		flatness_threshold:float=0.34,  
		flat_region_min_len:int=10) -> 'SignalRandomizer':
		
		def f(signal:'list[float]') -> 'list[float]':
			if len(signal) == 0:
				return signal

			new_sig = []
			flat_region_start = 0

			#helper function used to avoid code duplication
			def iteration(curr_data_point:float, i:int, end:bool):  #end is tru only the last iteration
				nonlocal flat_region_start
				if abs(curr_data_point - signal[i-1]) > flatness_threshold or end:  	#the signal is not flat
					flat_region_len = i - flat_region_start
					
					if flat_region_len < flat_region_min_len:   				#the flat region is too short
						new_sig.extend(signal[flat_region_start:i])
		
					else:     													#we found a flat region
						shift = shift_range.rand()
						new_sig.extend([j + shift for j in signal[flat_region_start:i]])
					
					flat_region_start = i
				
			for i in range(1, len(signal)):
				iteration(signal[i], i, False)
			
			iteration(signal[-1], len(signal), True) #check the last region

			return new_sig

		self._add(f)
		return self

	"""
	Add simmetric gaussian noise to the signal
	"""
	def add_noise(self, mean:float=0.1, stdv:float=0.2, minimum:float=0, maximum:float=0.2) -> 'SignalRandomizer':
		def nextGauss():
			x = random.gauss(mean, stdv)
			while x < minimum or x > maximum:
				x = random.gauss(mean, stdv)
			return x
		
		def randomSign():
			return random.choice([-1, 1])

		def f(signal:'list[float]') -> 'list[float]':
			return [i + nextGauss() * randomSign()  for i in signal]

		self._add(f)
		return self

	"""
	Add gaussian noise to the signal. Values can be clamped between minimum and maximum.
	Setting minimum and maximum to None (default) avoids clamping.
	
	"""
	def add_gaussian_noise(self, mean=0, stdv=0.4, minimum=None, maximum=None) -> 'SignalRandomizer':
		def nextGauss():
			x = random.gauss(mean, stdv)
			while x < minimum or x > maximum:
				x = random.gauss(mean, stdv)
			return x
		
		distribution = nextGauss
		if minimum is None or maximum is None:
			distribution = lambda: random.gauss(mean, stdv)
		
		def f(signal:'list[float]') -> 'list[float]':
			return [i + distribution()  for i in signal]

		self._add(f)
		return self

	"""
	Randomly stretches or shrinks the signal orizzontally.
	Resizes every window using a different ratio.
	"""
	def random_stretch(self, 
		window_size_range:IntRange=IntRange(10, 20),
		resize_ratio_range:FloatRange=FloatRange(0.7, 1.3)
		)  -> 'SignalRandomizer':
		def f(signal:'list[float]') -> 'list[float]':
			new_sig = []
			i = 0
			while i < len(signal):
				window = window_size_range.rand()
				if window <= 0:
					raise ValueError("window_size_range.rand() must be positive")
				
				ratio = resize_ratio_range.rand()
				if ratio <= 0:
					raise ValueError("resize_ratio_range.rand() must be positive")
					
				slice = SignalRandomizer._resize(signal[i:i+window], ratio=ratio, length=None)
				new_sig.extend(slice)
				i+= window

			return new_sig

		self._add(f)
		return self

	"""
	Resizes a signal to a given length or with a ratio
	"""  
	def resize(self, length=None, ratio=None)  -> 'SignalRandomizer':
		assert (length is not None and ratio is None) or (length is None and ratio is not None), "You must specify either length or ratio"
		
		def f(signal:'list[float]') -> 'list[float]':
			return SignalRandomizer._resize(signal, length, ratio)

		self._add(f)
		return self

	"""
	Resize a signal either to a length or using a ratio.
	If length is not None, then len(output) = length
	If ratio is not None, then len(outpt) = round(len(signal)*ratio)
	"""
	def _resize(signal, length=None, ratio=None):
		assert (length is not None and ratio is None) or (length is None and ratio is not None), "You must specify either length or ratio"
		
		if ratio is None:
			ratio = len(signal)/length
		else:
			length = round(len(signal)/ratio)
		
		if length == len(signal):
			return [i for i in signal]

		res = [0]*length
		for i in range(len(res)):
			x = i * ratio
			decimalLeft = x - math.floor(x)
			decimalRight = 1 - decimalLeft
			res[i] = (1 - decimalLeft) * signal[math.floor(x)] + (1 - decimalRight) * signal[math.ceil(x) if math.ceil(x) < len(signal) else len(signal) -1 ]

		return res



	def _add(self, p) -> None:
		self.assertNotBuilt()
		self.pipe.append(p)

	def assertNotBuilt(self) -> None:
		assert not self.built, "This signal randomizer has already been built"

	def build(self) -> None:
		self.built = True

	def randomize(self, signal:'list[float]') -> 'list[float]':
		assert self.built, "You must build the signal randomizer before randomizing"
		for p in self.pipe:
			signal = p(signal)
		return signal




def printOutputFile(s: "list[ScrappySequence]", filename: str):
    file = open(filename, "w")

    for i, v in enumerate(s):
        file.write(f"#Seq[{i + 1}],,,,\n")
        file.write("pos,base,current,sd,dwell\n")
        for j, w in enumerate(v):
            file.write(f"{j},{w.base},{w.current},{w.sd},{w.dwell}\n")

    file.close()



if __name__ == "__main__":

    sequences = parseScrappieOutput(
        "D:/SRP_DNAencoding/SRP-Synthetic-DNA-storage/test-alphabets/real_alphabets_F16.txt"
    )

    # tolerance = 4

    # for i in range(len(sequences)):
    # 	sequences[i] = removeInitial6As(sequences[i])
    # 	sequences[i] = randomizeSeparator(sequences[i], tolerance)
    # 	#sequences[i] = randomizeSequence(sequences[i])

    printOutputFile(
        sequences,
        "D:/SRP_DNAencoding/SRP-Synthetic-DNA-storage/test-alphabets/real_alphabets_F16.csv",
    )
    # printOutputFile(sequences, "output-separator8A-randomizedSequence.txt")
