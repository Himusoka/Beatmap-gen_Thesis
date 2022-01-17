import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from scipy.signal import argrelextrema
from sklearn.metrics import f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import glob
import librosa
from tqdm import tqdm

from utilities.feature_extractor import FeatureExtractor, convert_time

VAL_SIZE = 0.15
BATCH_SIZE = 256
EPOCHS = 15
PATIENCE = 5
LR_RATE = 0.005

class TrainingDataset(Dataset):
	def __init__(self, file, fuzzy_label=True, cache=None):
		self.extractor = FeatureExtractor()
		data, diffs, onsets = self.extractor.load_data(file, fuzzy_label, cache)

		self.x = torch.from_numpy(np.array(data, dtype=float))
		self.y = torch.from_numpy(diffs)
		self.z = torch.from_numpy(onsets)

		assert self.x.shape[0] == self.z.shape[0]
		assert self.x.shape[0] == self.y.shape[0]
		self.samples = self.x.shape[0]

	def __getitem__(self, index):
		return self.x[index], self.y[index], self.z[index]

	def __len__(self):
		return self.samples


class AudioDataset(Dataset):
	def __init__(self, file, diff, segments):
		self.extractor = FeatureExtractor()
		mels = self.extractor.extract_mel(file)
		data, diffs = self.extractor.convert_mel_to_in_data(mels, diff, intensity=segments)
		self.x = torch.from_numpy(np.array(data))
		self.y = torch.from_numpy(diffs)
		assert self.x.shape[0] == self.y.shape[0]
		self.samples = self.x.shape[0]

	def __getitem__(self, index):
		return self.x[index], self.y[index]

	def __len__(self):
		return self.samples


class CBlstm(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 10, (3, 7))
		self.conv2 = nn.Conv2d(10, 20, 3)
		self.norm = nn.BatchNorm2d(20)

		self.lstm1 = nn.LSTM(input_size=166, hidden_size=128, batch_first=True, bidirectional=True)
		self.tanh = nn.Tanh()

		self.fc1 = nn.Linear(7*2*128, 128)
		self.out = nn.Linear(128, 1)

		self.drop = nn.Dropout(p=0.8)
		self.sig = nn.Sigmoid()

	def forward(self, x, y):
		x = F.dropout(F.relu(self.conv1(x)), p=0.2, training=self.training)
		x = F.max_pool2d(x, (3,1))
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, (3,1))

		x = self.norm(x)
		x = x.permute(0, 3, 2, 1)
		x = torch.flatten(x, start_dim=2)
		y = torch.repeat_interleave(y, x.shape[1], dim=1).view(-1, 7, 6)
		x = torch.cat((x, y), 2)

		x, _ = self.lstm1(x)
		x = self.tanh(x)
		x = self.drop(torch.flatten(x, start_dim=1))

		x = self.drop(F.relu(self.fc1(x)))
		x = self.out(x)

		if not self.training:
			x = self.sig(x)

		return x

	def start_training(self, dir, device, outputdir="..\\models\\default", ev_set=None, file_set=None, cache=None):
		if not os.path.exists(outputdir):
			os.mkdir(outputdir)
		all_files = [f for f in glob.glob(os.path.join(dir, "**/*.osu"), recursive=True)]
		eval_files_len = int(len(all_files) * VAL_SIZE) + 1
		folders = glob.glob(os.path.join(dir, "*\\"))
		np.random.shuffle(folders)
		eval_files = []
		i = 0
		while len(eval_files) < eval_files_len:
			eval_files.extend([f for f in glob.glob(os.path.join(folders[i], "*.osu"))])
			i += 1
		files = [x for x in all_files if x not in eval_files]
		np.random.shuffle(files)
		files = files[:-eval_files_len]
		if ev_set is not None and file_set is not None:
			eval_files = np.load(ev_set)
			files = np.load(file_set)
		optimizer = optim.Adam(self.parameters(), lr=LR_RATE)
		#optimizer = optim.SGD(self.parameters(), lr=LR_RATE, momentum=0.85)
		loss_fn = nn.BCEWithLogitsLoss()

		loss_vals = []
		val_losses = []
		f_scores = []
		highest_f = 0
		prev_val_loss = float('inf')
		prev_state = self.state_dict()
		model_thresh = 0
		training_patience = PATIENCE

		for epoch in range(EPOCHS):
			self.train()
			np.random.shuffle(files)
			running_loss = 0
			dataset_len = 0
			for i, file in enumerate(files):
				try:
					dataset = TrainingDataset(file, cache=cache)
					loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
					dataset_len += len(loader)
					print("Epoch: " + str(epoch) + "/" + str(EPOCHS) + ", data: " + str(i) + "/" + str(len(files)))
					for (batch_X, batch_Y, batch_Z) in tqdm(loader):
						optimizer.zero_grad()
						out_on = self(batch_X.to(device, dtype=torch.float), batch_Y.to(device, dtype=torch.float))
						loss = loss_fn(out_on.view(-1), batch_Z.to(device, dtype=torch.float))
						loss.backward()
						optimizer.step()
						running_loss += loss.item()
				except FileNotFoundError as e:
					print(str(e))
					files.remove(file)

			train_loss = running_loss/dataset_len
			print("loss: ", train_loss)
			loss_vals.append(train_loss)
			val_loss, f1, thresh = self.evaluate(eval_files, device)
			if prev_val_loss < val_loss:
				print("loss increased", abs(training_patience - 5))
				training_patience -= 1
				if training_patience == -1:
					print("Early training stop checkpoint after", epoch, "epochs")
					torch.save(prev_state, os.path.join(outputdir, "onset_model_check.pth"))
					np.save(os.path.join(outputdir, "onset_check_thresh.npy"), np.array(model_thresh))
			else:
				prev_state = self.state_dict()
				training_patience = PATIENCE
				model_thresh = thresh
				prev_val_loss = val_loss
			f_scores.append(f1)
			val_losses.append(val_loss)
			if f_scores[-1] > highest_f:
				np.save(os.path.join(outputdir, "onset_thresh_best_f1.npy"), np.array(thresh))
				torch.save(self.state_dict(), os.path.join(outputdir, "onset_model_best_f1.pth"))
				highest_f = f_scores[-1]

		np.save(os.path.join(outputdir, "onset_thresh.npy"), np.array(thresh))
		np.save(os.path.join(outputdir, "train_files.npy"), np.array(files))
		np.save(os.path.join(outputdir, "val_files.npy"), np.array(eval_files))
		torch.save(self.state_dict(), os.path.join(outputdir, "onset_model.pth"))
		return loss_vals, val_losses, f_scores

	def wide_window(self, onsets, ground, window_size=2):
		windowed_onset = []
		windowed_ground = []

		for i in range(window_size, onsets.shape[0] - window_size):
			if np.count_nonzero(onsets[i - window_size:i + window_size + 1]) > 0:
				windowed_onset.append(1)
			else:
				windowed_onset.append(0)
			if np.count_nonzero(ground[i - window_size:i + window_size + 1]) > 0:
				windowed_ground.append(1)
			else:
				windowed_ground.append(0)
		return windowed_onset, windowed_ground


	def evaluate(self, files, device, dir=None, model=None):
		if model is not None:
			self.load_state_dict(torch.load(os.path.join(model, "onset_model.pth"), map_location=device))
		if dir is not None:
			files = [f for f in glob.glob(os.path.join(dir, "**/*.osu"), recursive=True)]
		loss_fn = nn.BCEWithLogitsLoss()
		ground = []
		diffs = []
		running_loss = 0
		dataset_len = 0
		with torch.no_grad():
			self.eval()
			predictions = []
			for i, file in tqdm(enumerate(files)):
				print("Data: " + str(i) + "/" + str(len(files)))
				try:
					dataset = TrainingDataset(file, fuzzy_label=False)
					loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)
					dataset_len += len(loader)
					for (batch_X, batch_Y, batch_Z) in tqdm(loader):
						out_0 = self(batch_X.to(device, dtype=torch.float), batch_Y.to(device, dtype=torch.float))
						loss = loss_fn(out_0.view(-1), batch_Z.to(device, dtype=torch.float))
						out_0 = self.sig(out_0)
						predictions.extend(out_0.cpu())
						diffs.extend(torch.argmax(batch_Y, dim=1).cpu())
						ground.extend(batch_Z.view(-1).cpu())
						running_loss += loss.item()
				except FileNotFoundError as e:
					print(str(e))
					files.remove(file)
			predictions = np.array(predictions)
			predictions_smooth = np.convolve(predictions, np.hamming(5), 'same')

			pr, re, thresh =  precision_recall_curve(ground, predictions)
			fscore = (2*pr*re)/(pr+re)
			ix = np.argmax(fscore)
			print("Best:", thresh[ix], "fscore1:", fscore[ix])
			#plt.plot([0,1],[0,1], linestyle="--")
			#plt.plot(pr, re, marker='.')
			#plt.scatter(pr[ix], re[ix], marker='o', color="black")
			#plt.show()
			maxima = argrelextrema(predictions_smooth, np.greater_equal, order=1)[0]
			pred_maxed = np.zeros(len(predictions))
			for i in maxima:
				if predictions[i] >= (thresh[ix] - diffs[i] * 0.015):
					pred_maxed[i] = 1

		pred_maxed, ground = self.wide_window(pred_maxed, ground)
		val_loss = running_loss/dataset_len
		onset_f = f1_score(ground, pred_maxed)
		print("f1_score:", onset_f)
		return val_loss, onset_f, thresh[ix]

	def _calc_density(self, onsets, tempo):
		if not onsets or len(onsets) == 1:
			return 0
		onsets = librosa.frames_to_time(onsets, sr=44100) * 1000
		avg = 0
		prev = onsets[0]
		curr_tempo = -1
		for onset in onsets[1:]:
			if curr_tempo + 1 < tempo.shape[0]:
				if onset >= tempo[curr_tempo + 1][0]:
					curr_tempo += 1
			avg += convert_time(onset - prev, tempo[curr_tempo - 1][1])
			prev = onset
		return avg / (len(onsets) - 1)

	def infer(self, file, difficulty, segments, tempo, device, model="..\\models\\default"):
		self.load_state_dict(torch.load(os.path.join(model, "onset_model.pth"), map_location=device))
		thresh = np.load(os.path.join(model, "onset_thresh.npy"))
		predictions = []
		with torch.no_grad():
			self.eval()
			dataset = AudioDataset(file, difficulty, segments)
			dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
			for (x, y) in tqdm(dataloader):
				out_0 = self(x.to(device, dtype=torch.float), y.to(device, dtype=torch.float))
				out_0 = self.sig(out_0)
				predictions.extend(out_0.cpu())

		predictions_smooth = np.convolve(predictions, np.hamming(5), 'same')
		maxima = argrelextrema(predictions_smooth, np.greater_equal, order=1)[0]
		avg_density = 0
		j = - difficulty
		while avg_density >= 8 or avg_density == 0:
			hits = []
			for i in maxima:
				if predictions[i] >= (thresh + j * 0.015):
					hits.append(i)
			avg_density = self._calc_density(hits, tempo)
			print(avg_density)
			j -= 1
		print(len(hits))

		return librosa.frames_to_time(hits, sr=44100) * 1000
