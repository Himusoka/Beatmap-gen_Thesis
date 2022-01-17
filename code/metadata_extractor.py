import os
import csv
import math
import argparse

import glob
from numpy.lib.function_base import extract
from tqdm import tqdm
import librosa
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from utilities.beatmap import Beatmap
from utilities.osu_parser import OsuParser
from utilities.feature_extractor import convert_diff


class MetadataExtractor():
	p = OsuParser()

	def __init__(self, files):
		self.fig_names = files.rsplit("\\", 1)[-1]
		self.files = [f for f in glob.glob(os.path.join(files, "**/*.osu"), recursive=True)]
		self.n_audios = 0
		self.n_beatmaps = len(self.files)
		self.avg_beatmaps_to_audio = 0

		self.total_frames = 0
		self.avg_frames = 0

		self.objects = {
			"Circle" : 0,
			"Slider" : 0,
			"Reverse" : 0,
			"Spinner" : 0}
		self.total_obj = 0

		self.avg_objects_to_frames = 0
		
		self.avg_tempo = 0
		self.mean_tempo = 0
		self.std_dev_tempo = 0
		self.highest_tempo = 0
		self.lowest_tempo = math.inf

		self.avg_diff = 0
		self.mean_diff = 0
		self.std_dev_diff = 0

		self.mean_pattern = 0
		self.avg_pattern_len = 0
		self.std_dev_pattern = 0
		self.longest_pattern = 0
		self.pattern_hist = []

	def get_data(self):
		new_combo = 0
		n_beatmaps = 0
		pattern_len = 0
		beatmaps_to_audio = 0
		objects_per_frame = 0
		tempo = []
		patterns = []
		difficulty = []
		audio_files = set()

		for file in tqdm(self.files):
			bm = self.p.parse_map(file)
			dir = bm.path.split('\\')[-2]
			if dir in audio_files:
				n_beatmaps += 1
			else:
				audio_files.add(dir)
				beatmaps_to_audio += n_beatmaps
				n_beatmaps = 1
			file_len = librosa.get_duration(filename=os.path.join(file.rsplit('\\', 1)[0], bm.general["AudioFilename"]))
			file_frames = librosa.core.time_to_frames(file_len, sr=44100)
			self.total_frames += file_frames

			self.objects["Circle"] += bm.additional["nCircle"]
			self.objects["Slider"] += bm.additional["nSlider"]
			self.objects["Reverse"] += bm.additional["nReverse"]
			self.objects["Spinner"] += bm.additional["nSpinner"]
			self.total_obj += bm.additional["nObjects"]

			objects_per_frame += bm.additional["nObjects"] / file_frames

			tempo.append(bm.additional["BPM"])

			difficulty.append(convert_diff(bm.additional["starRating"]))
			for hit_object in bm.hit_objects:
				if hit_object["type"] & 0x04 != 0 or hit_object == bm.hit_objects[-1]:
					new_combo += 1
					if pattern_len != 0:
						patterns.append(pattern_len)
					pattern_len = 0
				pattern_len += 1

		self.avg_frames = self.total_frames / self.n_beatmaps
		self.n_audios = len(audio_files)
		self.avg_beatmaps_to_audio = beatmaps_to_audio / self.n_audios

		self.avg_objects_to_frames = objects_per_frame / self.n_beatmaps

		tempo = np.array(tempo)
		self.avg_tempo = np.average(tempo)
		idx = tempo.shape[0] // 2
		if tempo.shape[0] % 2 != 0 and tempo.shape[0] > 1:
			self.mean_tempo = (tempo[idx - 1] + tempo[idx]) / 2
		else:
			self.mean_tempo = tempo[idx]
		self.std_dev_tempo = np.std(tempo)
		self.lowest_tempo = tempo.min()
		self.highest_tempo = tempo.max()	
	
		difficulty = np.array(difficulty)
		self.avg_diff = np.average(difficulty)
		idx = difficulty.shape[0] // 2
		if difficulty.shape[0] % 2 != 0 and difficulty.shape[0] > 1:
			self.mean_diff = (difficulty[idx - 1] + difficulty[idx]) / 2
		else:
			self.mean_diff = difficulty[idx]
		self.std_dev_diff = np.std(difficulty)

		patterns = np.array(patterns)
		idx = patterns.shape[0] // 2
		if patterns.shape[0] % 2 != 0 and patterns.shape[0] > 1:
			self.mean_pattern = (patterns[idx - 1] + patterns[idx]) / 2
		else:
			self.mean_pattern = patterns[idx]
		self.std_dev_pattern = np.std(patterns)
		self.avg_pattern_len = np.average(patterns)
		self.longest_pattern = patterns.max()
		
		self.pattern_hist = np.histogram(patterns, bins=np.arange(1, self.longest_pattern + 2))

		plot1 = plt.figure(1)
		ax = plot1.add_subplot(1,1,1)
		plt.hist(difficulty, bins=np.arange(0, 6), density=True, stacked=True, align="left")
		plt.xticks(np.arange(0, 6, 1))
		ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
		plt.xlabel("Difficulty")
		plt.ylabel("Percent")
		plt.draw()
		plt.savefig(os.path.join("..\\metadata", self.fig_names + "_diff.png"))
		plt.clf()

		plot2 = plt.figure(2)
		ax = plot2.add_subplot(1,1,1)
		plt.hist(patterns, bins=np.arange(1, self.longest_pattern + 1), density=True, stacked=True, align="left")
		ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
		plt.xticks(np.arange(1, self.longest_pattern + 1, 1))
		plt.xlabel("Pattern Length")
		plt.ylabel("Percent")
		plt.draw()
		plt.savefig(os.path.join("..\\metadata", self.fig_names + "_pattern.png"))
		plt.clf()
	


		
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("f", help="the dataset path")
	args = parser.parse_args()

	if not os.path.exists("..\\metadata"):
		os.makedirs("..\\metadata")
	extractor = MetadataExtractor(args.f)
	extractor.get_data()
	with open(os.path.join("..\\metadata", args.f.rsplit('\\', 1)[-1] + "_metadata.csv"), 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["Audio files", "beatmaps", "avg bm to audio", "total frames", "avg_frames in audio"])
		writer.writerow([extractor.n_audios, extractor.n_beatmaps, extractor.avg_beatmaps_to_audio, extractor.total_frames, extractor.avg_frames])
		writer.writerow(["circle", "slider", "reverse", "spinner", "total", "avg obj to frames"])
		writer.writerow([
			extractor.objects["Circle"], 
			extractor.objects["Slider"], 
			extractor.objects["Reverse"], 
			extractor.objects["Spinner"],
			extractor.total_obj,	 
			extractor.avg_objects_to_frames])
		writer.writerow(["avgTempo", "median tempo", "std dev tempo", "highest tempo", "lowest tempo"])
		writer.writerow([extractor.avg_tempo, extractor.mean_tempo, extractor.std_dev_tempo, extractor.highest_tempo, extractor.lowest_tempo])
		writer.writerow(["avg diff", "median diff", "std dev diff"])
		writer.writerow([extractor.avg_diff, extractor.mean_diff, extractor.std_dev_diff])
		writer.writerow(["avg pattern len", "median pattern len", "std dev pattern", "longest pattern"])
		writer.writerow([extractor.avg_pattern_len, extractor.mean_pattern, extractor.std_dev_pattern, extractor.longest_pattern])
		writer.writerow(["pattern histogram"])
		writer.writerow(extractor.pattern_hist[1])
		writer.writerow(extractor.pattern_hist[0])



	
