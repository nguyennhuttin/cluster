import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
import math
import urllib.request
import json

"""
ScrappySequenceItem represents a single item in a sequence returned by scrappy
"""
class ScrappySequenceItem:
	def __init__(self, base:str, current:float, sd:float, dwell:float):
		assert len(base) == 1 and type(current) == float and type(sd) == float and type(dwell) == float
		self.base = base
		self.current = current
		self.sd = sd
		self.dwell = dwell
	
	def __str__(self):
		return "Base: {}, Current: {}, SD: {}, Dwell: {}".format(self.base, self.current, self.sd, self.dwell)

"""
Signal represents a signal corresponding to a DNA sequence, which was generated using 
a ScrappySequence
"""
class Signal:
	def __init__(self, signal:'list[float]', indexes:'list[int]', sequence:'ScrappySequence'):
		self.signal = signal
		self.indexes = indexes
		self.sequence = sequence

	"""
	Return the sequence item that generated the signal vlue at position item in the
	signal array.
	""" 
	def __getitem__(self, item):
		i = self.indexes[item]
		return self.sequence[i]

	def __len__(self):
		return len(self.indexes)

"""
ScrappySequence represents a DNA sequence that was generated using a Scrappy.
"""
class ScrappySequence:
	def __init__(self, bases:'list[str]', currents:'list[float]', sds:'list[float]', dwells:'list[float]'):
		assert len(bases) == len(currents) == len(sds) == len(dwells)
		
		self.bases = bases 
		self.currents = currents
		self.sds = sds
		self.dwells = dwells
	
	@staticmethod
	def fromSequenceItems(items:'list[ScrappySequenceItem]'):
		s = ScrappySequence([], [], [], [])
		for i in items:
			s.extend(i.base, i.current, i.sd, i.dwell)
		return s

	"""
	Add a signal to the sequence.
	"""
	def extend(self, base:str, current:float, sd:float, dwell:float):
		self.bases.append(base)
		self.currents.append(current)
		self.sds.append(sd)
		self.dwells.append(dwell)

	"""
	Return the ScrappySequenceItem at this index
	"""
	def __getitem__(self, i):
		return ScrappySequenceItem(
			self.bases[i],
			self.currents[i],
			self.sds[i],
			self.dwells[i])

	def slice(self, a, b):
		return [self[i] for i in range(a, b)]

	def __len__(self):
		return len(self.bases)
	
	def __iter__(self):
		return (self[i] for i in range(len(self)))

	def __str__(self):
		out = []
		for i, v in enumerate(self):
			out.append("{} {} {} {} {}".format(i, v.base, v.current, v.sd, v.dwell))
		return "\n".join(out)
		
	"""
	Generate a random Signal corresponging to this ScrappySequence
	"""
	def generateRandomSignal(self):
		s = Signal([], [], self)
		for i, v in enumerate(self):
			p = 1/v.dwell
			if p > 1:
				p = 1
			r = np.random.geometric(p=p) + 1
			signals = v.sd *np.random.normal(size= r) + v.current
			indexes = i*np.ones(r)
			s.signal.extend(signals)
			s.indexes.extend(indexes)

		return s

	"""
	Generate the expected signal correspoding to this ScrappySequence.
	"""
	def generateExpectedSignal(self):
		signal = Signal([], [], self)
		for i, v in enumerate(self):
			samples = round(v.dwell)
			for j in range(samples):
				signal.signal.append(v.current)
				signal.indexes.append(i)
			
		return signal


"""
ScrappySequenceGenerator is a class that can generate a ScrappySequence from a given
sequence.
"""

"""
Parses a file returned by Scrappy squiggle
"""
def parseScrappieOutput(filename) -> 'list[ScrappySequence]': 
	
	INDEX = 0
	BASE = 1
	CURRENT = 2
	SD = 3
	DWELL = 4
	
	sequences:list[ScrappySequence] = []

	with open(filename, 'r', encoding='utf-8') as file:
		for line in file:
			if line.startswith("#"):
				sequences.append(ScrappySequence([], [], [], []))
				continue
			
			if line[0].isalpha():
				continue
			
			else:
				pieces = list(filter(lambda x: x != "", line.split("\t")))
				pieces = list(map(lambda x: x.strip(), pieces))
				sequences[len(sequences) - 1].extend(pieces[BASE], float(pieces[CURRENT]), float(pieces[SD]), float(pieces[DWELL]))
	
	return sequences

def isScrappyInterfaceOnline():
	try:
		x = urllib.request.urlopen("http://localhost:7777/").read()
		return json.loads(x) == "Hello Scrappy!"
	except:
		print("Could not contact scrappy interface")
		return False

def getScrappySequence(seq:str)->ScrappySequence:
	x = urllib.request.urlopen("http://localhost:7777/", data=seq.encode('utf-8')).read()
	table = json.loads(x)
	for i, v in enumerate(table):
		print(seq[i], v[0], v[1], v[2])

	bases = [i.upper() for i in seq]
	current = [float(i[0]) for i in table]
	sd = [float(i[1]) for i in table]
	dwell = [float(i[2]) for i in table]
	return ScrappySequence(bases, current, sd, dwell)
		

if __name__ == "__main__":	
	assert isScrappyInterfaceOnline()

	sequences = [
		#parseScrappieOutput("output-separator8A.txt")[0],
		getScrappySequence("AAAAAAAAAAAAAACCCCCCCCTTTTTTTTAAAAAAAAATTTTTTTTTTAAAAAAAAAATTTTTTTGGGGGGGGAAAAAAAAAAAAAA")
	]	

	for s in sequences:
	
		
		#s = parseScrappieOutput(filename)[0]
		#s = getScrappySequence("AACCGCTTTTCGGCCCCGGGGGAAAGGAGGAGATTTCCC")

		fig, axs = plt.subplots(2, figsize=(10, 6))

		x = []
		y = []
		time = 0

		def addPoint(xv, yv):
			x.append(xv)
			y.append(yv)

		addPoint(time, s[0].current)
		for i in range(len(s) - 1):
			middleX = (time*2 + s[i].dwell) / 2 
			axs[0].text(middleX, s[i].current + 0.03, s[i].base, ha='center', va='center', fontsize=10, path_effects=[patheffects.withStroke(linewidth=2, foreground='w')])
			time += s[i].dwell
			addPoint(time, s[i].current)
			addPoint(time, s[i+1].current)

		axs[0].plot(x, y, label="Line")
		plt.ylim((-2, 2))

		signal = s.generateRandomSignal()
		signal_time = [i for i in range(len(signal))]
		axs[1].plot(signal_time, signal.signal, label="Line")
		
		expected_signal = s.generateExpectedSignal()
		expected_signal_time = [i for i in range(len(expected_signal))]
		axs[1].plot(expected_signal_time, expected_signal.signal, label="Line", color="red")

	plt.show()
	
