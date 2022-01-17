import os
import math
import argparse
from collections import deque

import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import scipy.signal as sp
import matplotlib.pyplot as plt

from utilities.osu_parser import OsuParser
from utilities.beatmap import Beatmap


def convert_time(length, bpm):
	dur = round(1 / (bpm / length), 3)
	if dur <= 0.30:
		dur = 0.25
	elif dur <= 0.55:
		dur = 0.5
	elif dur <= 0.80:
		dur = 0.75
	elif dur <= 1.05:
		dur = 1.0
	return dur

def convert_diff(difficulty):
	if difficulty < 2:
		converted_diff = 0
	elif difficulty < 2.7:
		converted_diff = 1
	elif difficulty < 4:
		converted_diff = 2
	elif difficulty < 5.3:
		converted_diff = 3
	elif difficulty < 6.5:
		converted_diff = 4
	else:
		converted_diff = 5
	return converted_diff

class FeatureExtractor():
	p = OsuParser()
	sr = 44100
	x = None

	def __init__(self, stride=512, extraframes=7):
			self.STRIDE = stride
			self.EXTRAFRAMES = extraframes

	def _fuzzyLabel(self, x):
		return math.exp(-((x**2)/(2*3)))

	def extract_mel(self, audio_path):
		if self.x is None:
			self.x, sr = librosa.load(audio_path, sr=self.sr)
		mels = []
		for n_fftpower in tqdm(range(10, 13)):
			S = np.abs(librosa.core.stft(self.x, n_fft=2**n_fftpower, hop_length=self.STRIDE, window=sp.windows.hamming(2**n_fftpower)))**2
			M = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=80, fmin=27.5, fmax=16000.0)
			M = librosa.power_to_db(M + 1e-16)
			for i in range(self.EXTRAFRAMES):
				M = np.append(M, np.zeros((M.shape[0], 1)), axis=1)
				M = np.insert(M, 0, 0, axis=1)
			mels.append(M.T)
		mels = np.array(mels).swapaxes(0, 1)
		return mels

	def get_intensity(self, audio_path, segment_times, exagerate=False):
		if self.x is None:
			self.x, self.sr = librosa.load(audio_path, sr=self.sr)
		S = librosa.core.stft(self.x, hop_length=self.STRIDE)
		energy = librosa.feature.rms(S=S, hop_length=self.STRIDE)
		if exagerate:
			energy *= energy
		intensity = []
		lowest = math.inf
		highest = -math.inf
		for i, t in enumerate(segment_times[0]):
			flag = 0
			times = librosa.time_to_frames(t, sr=self.sr, hop_length=self.STRIDE)
			avg = np.average(energy[0][times[0]:times[1]])
			if avg < lowest:
				lowest = avg
				flag = -1
			elif avg > highest:
				highest = avg
				flag = 1
			if "chorus" in segment_times[1][i]:
				chorus = 1
			else:
				chorus = 0
			intensity.append([t[0] * 1000, flag, avg, chorus])
		return np.array(intensity), np.average(energy[0])

	def get_slider_len(self, object, bm):
		slider_len = float(object["objectParams"]["length"])
		slider_mult = bm.difficulty["SliderMultiplier"]
		time = object["time"]
		red_time_point = [y for y in bm.timing_points if y["uninherited"]]
		green_time_point = [y for y in bm.timing_points if not y["uninherited"]]
		beat_length = red_time_point[0]["beatLength"]
		prev = red_time_point[0]
		for point in red_time_point:
			if time < point["time"]:
				beat_length = prev["beatLength"]
				break
			prev = point
		if green_time_point:
			prev_mult = 100
			for point in green_time_point:
				if time < point["time"]:
					slider_mult *= (100 / prev_mult)
					break
				elif point == green_time_point[-1]:
					slider_mult *= (100 / (point["beatLength"]))
					break
				prev_mult = -prev["beatLength"]
		return slider_len / (slider_mult * 100) * beat_length


	def extract_given_onsets(self, bm, n_frames, fuzzy_label=True):
		onsets = np.zeros(n_frames, dtype=float)
		n_combos = np.zeros(n_frames, dtype=float)
		hit_object = []
		for object in bm.hit_objects:
			hit_object.append(librosa.core.time_to_frames(object["time"] / 1000, sr=self.sr, hop_length=self.STRIDE))
			if object["type"] & 0x02 != 0:
				length = self.get_slider_len(object, bm)
				for repeat in range(int(object["objectParams"]["slides"])):
					hit_object.append(librosa.core.time_to_frames((object["time"] + length)/ 1000, sr=self.sr, hop_length=self.STRIDE))
			elif object["type"] & 0x08 != 0:
				hit_object.append(librosa.core.time_to_frames(object["objectParams"] / 1000, sr=self.sr, hop_length=self.STRIDE))

		if fuzzy_label:
			for i in hit_object:
				for j in range(4):
					if i - j > 0:
						if onsets[i - j] <= self._fuzzyLabel(j):
							onsets[i - j] = self._fuzzyLabel(j)
					if i + j < onsets.shape[0]:
						if onsets[i + j] <= self._fuzzyLabel(j):
							onsets[i + j] = self._fuzzyLabel(j)
				onsets[i] = 1
		else:
			for i in hit_object:
				onsets[i] = 1
		return onsets

	def convert_mel_to_in_data(self, mels, target_diff, add_diff=False, intensity=None):
		data = []
		diff = convert_diff(target_diff)
		difficulty = np.eye(6)[diff]
		for frame in tqdm(range(self.EXTRAFRAMES, mels.shape[0] - self.EXTRAFRAMES)):
			if intensity is not None:
				for (t, flag, _, _) in np.flip(intensity):
					if t < frame:
						if flag == -1 and diff != 0:
							diff -= 1
						elif flag == 1 and diff != 5:
							diff += 1
						else:
							diff = convert_diff(target_diff)
			frame_windows = []
			for window in range(mels.shape[1]):
				sequence = []
				for i in range(-self.EXTRAFRAMES, self.EXTRAFRAMES + 1):
					sequence.append(mels[frame + i][window])
				frame_windows.append(sequence)
			if add_diff:
				data.append([np.array(frame_windows).swapaxes(1, 2), difficulty])
			else:
				difficulty = np.vstack((difficulty, np.eye(6)[diff]))
				data.append(np.array(frame_windows).swapaxes(1, 2))
		return data, difficulty[:-1]

	def extract_types(self, file):
		# hit , slider, spinner
		bm = self.p.parse_map(file)
		# prev, diff, prevtime, nexttime
		# [tempo, diff, prev, new_combo, prevtime, nexttime] x3
		types = []
		data = []
		new_combos = []
		next_time = bm.hit_objects[0]["time"]
		const_feature = np.array(bm.timing_points[0]["beatLength"])
		dif = convert_diff(bm.additional["starRating"])
		red_time_point = [y for y in bm.timing_points if y["uninherited"]]
		cur_tempo_idx = 0

		const_feature = np.append(const_feature, np.eye(6)[dif])
		past_var_feat = deque([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, convert_time(next_time, red_time_point[cur_tempo_idx]["beatLength"])]], maxlen=3)
		for i, hit in enumerate(bm.hit_objects):
			if cur_tempo_idx + 1 < len(red_time_point) and hit["time"] >= red_time_point[cur_tempo_idx + 1]["time"]:
				cur_tempo_idx += 1
			type = hit["type"]
			n_comb = 0
			prev_time = convert_time(next_time, red_time_point[cur_tempo_idx]["beatLength"])
			if i + 1 == len(bm.hit_objects):
				next_time = 0.1
			else:
				next_time = bm.hit_objects[i + 1]["time"] - hit["time"]
			sample = []
			past_feature_list = list(past_var_feat)
			for j in range(3):
				frame = list(const_feature)
				frame.extend(past_feature_list[j])
				sample.append(frame)
			data.append(sample)
			if type & 0x04 != 0:
				n_comb = 1
				new_combos.append(1)
			else:
				new_combos.append(0)
			#slider
			if type & 0x02 != 0:
				length = self.get_slider_len(hit, bm)
				feature = np.array(np.eye(3)[1])
				feature = np.append(feature, [n_comb, prev_time, convert_time(length * int(hit["objectParams"]["slides"]), red_time_point[cur_tempo_idx]["beatLength"])])
				next_time = next_time - length * int(hit["objectParams"]["slides"])
				types.append(1)
				past_var_feat.append(feature)

			#spinner
			elif type & 0x08 != 0:
				length = hit["objectParams"] - hit["time"]
				feature = np.array(np.eye(3)[2])
				feature = np.append(feature, [n_comb, prev_time, convert_time(length, red_time_point[cur_tempo_idx]["beatLength"])])
				types.append(2)
				past_var_feat.append(feature)

			#hit circle
			else:
				feature = np.array(np.eye(3)[0])
				feature = np.append(feature, [n_comb, prev_time, convert_time(next_time, red_time_point[cur_tempo_idx]["beatLength"])])
				types.append(0)
			past_var_feat.append(feature)
		#print(data[0:3])
		return types, data[:-1], new_combos

	def convert_one(self, file, fuzzy_label):
		bm = self.p.parse_map(file)
		mels = self.extract_mel(os.path.join(os.path.dirname(file), bm.general["AudioFilename"]))
		print(mels.shape)
		data, diff = self.convert_mel_to_in_data(mels, bm.additional["starRating"])
		onsets = self.extract_given_onsets(bm, len(data), fuzzy_label=fuzzy_label)
		return data, diff, onsets

	def load_data(self, file, fuzzy_label, cache=None):
		if cache is None:
			data, difficulty, onsets = self.convert_one(file, fuzzy_label)
			return data, difficulty, onsets
		if not os.path.exists("..\\featureCache"):
			os.mkdir("..\\featureCache")
		filebase = os.path.join("..\\featureCache", file.split('\\')[-1].rsplit('.', 1)[0].split('[')[0].strip())
		filename = file.rsplit('.', 1)[-2].split('[')[-1].strip()
		try:
			print("Succesfully found data for", file)
			data = np.load(os.path.join(filebase, "_data.npy"))
			print(data.shape[0])
			difficulty = np.load(os.path.join(filebase, "[" + filename + "_diff.npy"))
			if fuzzy_label:
				onsets = np.load(os.path.join(filebase, "[" + filename + "_onsets.npy"))
			else:
				onsets= self.extract_given_onsets(self.p.parse_map(file), data.shape[0], fuzzy_label=fuzzy_label)
			print("Succesfully loaded data")
			return data, difficulty, onsets
		except IOError:
			print("No data found: building new data for ", file)
			data, difficulty, onsets = self.convert_one(file, fuzzy_label)
			if not os.path.exists(filebase):
				os.mkdir(filebase)
			np.save(os.path.join(filebase, "_data.npy"), data)
			np.save(os.path.join(filebase, "[" + filename + "_diff.npy"), difficulty)
			np.save(os.path.join(filebase, "[" + filename + "_onsets.npy"), onsets)
			return data, difficulty, onsets
