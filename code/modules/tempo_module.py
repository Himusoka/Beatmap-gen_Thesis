import numpy as np
from tempocnn.classifier import TempoClassifier
from tempocnn.feature import read_features

def get_tempo(file, queue=None):
	classifier = TempoClassifier('cnn')
	features = read_features(file, frames=256, hop_length=22)
	global_bpm = classifier.estimate_tempo(features, interpolate=False)

	local_tempo_classes = classifier.estimate(features)
	max_predictions = np.argmax(local_tempo_classes, axis=1)
	local_tempi = classifier.to_bpm(max_predictions)
	prev = local_tempi[0]
	local_bpm = np.array([[0, local_tempi[0]]])
	for i, tempo in enumerate(local_tempi):
		if abs(tempo - prev) > prev * 0.04:
			if abs(tempo - (2 * prev)) > prev * 0.04 and abs((2 * tempo) - prev) > prev * 0.04:
				local_bpm = np.vstack((local_bpm, [(i * 22 * 512 / 110)* 10, tempo]))
				prev = tempo
	if queue is not None:
		queue.put(global_bpm)
		queue.put(local_bpm)
	return global_bpm, local_bpm
