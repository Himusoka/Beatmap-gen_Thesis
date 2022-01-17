import os
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

from utilities.feature_extractor import FeatureExtractor, convert_time


BATCH_SIZE = 16
VAL_SIZE = 0.15
EPOCHS = 50
PATIENCE = 5
LR_RATE = 0.0005

class TypeDataset(Dataset):
	def __init__(self, file):
		self.extractor = FeatureExtractor()
		ground, data, comboground = self.extractor.extract_types(file)
		self.x = torch.from_numpy(np.array(data))
		self.y = torch.from_numpy(np.array(ground))
		self.z = torch.from_numpy(np.array(comboground))
		self.samples = self.x.shape[0]

	def __getitem__(self, index):
		return self.x[index].float(), self.y[index].long(), self.z[index].float()

	def __len__(self):
		return self.samples


class LstmClustering(nn.Module):
	def __init__(self):
		super().__init__()
		self.lstm1 = nn.LSTM(input_size=13, hidden_size=128, batch_first=True, num_layers=2)

		self.lin = nn.Linear(3*128, 128)
		self.out = nn.Linear(128, 3)

		self.clu = nn.Linear(3*128, 256)
		self.clu2 = nn.Linear(256, 128)
		self.cluout = nn.Linear(128, 1)

		self.sig = nn.Sigmoid()
		self.soft = nn.Softmax(dim=1)

	def forward(self, x, h_t=None, c_t=None):
		if h_t is None or c_t is None:
			x, (h_n, c_n) = self.lstm1(x)
		else:
			x, (h_n, c_n) = self.lstm1(x, (h_t, c_t))
		x = F.relu(x)
		lstmout = torch.flatten(x, start_dim=1)
		x1 = F.dropout(F.relu(self.lin(lstmout)), training=self.training)
		x1 = self.out(x1)

		x2 = F.dropout(F.relu(self.clu(lstmout)), training=self.training)
		x2 = self.cluout(F.relu(self.clu2(x2)))

		if not self.training:
			x1 = self.soft(x1)
			x2 = self.sig(x2)

		return x1, x2, h_n, c_n

	def start_training(self, dir, device, outputdir="..\\models\\default", ev_set=None, file_set=None):
		if not os.path.exists(outputdir):
			os.mkdir(outputdir)
		modelname = dir.split('\\')[-1]
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
		if ev_set is not None and file_set is not None:
			eval_files = np.load(ev_set)
			files = np.load(file_set)

		optimizer = optim.Adam(self.parameters(), lr=LR_RATE)
		loss_fn1 = nn.CrossEntropyLoss()
		loss_fn2 = nn.BCEWithLogitsLoss()

		loss_vals = []
		val_losses = []
		highest_f = 0
		loss_vals = []
		f_scores = []
		prev_val_loss = float('inf')
		prev_state = self.state_dict()
		model_thresh = 0
		training_patience = PATIENCE

		for epoch in range(EPOCHS):
			self.train()
			running_loss = 0
			dataset_len = 0
			np.random.shuffle(files)
			for i, file in enumerate(files):
				try:
					dataset = TypeDataset(file)
					loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)
					dataset_len += len(loader)
					print("Epoch: " + str(epoch) + "/" + str(EPOCHS) + ", data: " + str(i) + "/" + str(len(files)))
					for (batch_X, batch_Y, batch_Z) in tqdm(loader):
						optimizer.zero_grad()
						out1, out2, _, _ = self(batch_X.to(device))

						loss1 = loss_fn1(out1.view(-1, 3), batch_Y.to(device))
						loss2 = loss_fn2(out2.view(-1), batch_Z.to(device))
						loss = loss1 + loss2
						loss.backward()
						optimizer.step()
						running_loss += loss.item()
				except FileNotFoundError as e:
					print(str(e))
					files.remove(file)

			train_loss = running_loss/dataset_len
			print("loss: ", train_loss)
			loss_vals.append(train_loss)
			val_loss, f1, thresh, _ = self.evaluate(eval_files, device)
			if prev_val_loss < val_loss:
				print("loss increased", abs(training_patience - 5))
				training_patience -= 1
				if training_patience == -1:
					print("Early training stop checkpoint after", epoch, "epochs")
					torch.save(prev_state, os.path.join(outputdir, "seq_clust_model_check.pth"))
					np.save(os.path.join(outputdir, "seq_clust_thresh.npy"), np.array(model_thresh))
			else:
				prev_state = self.state_dict()
				training_patience = PATIENCE
				model_thresh = thresh
				prev_val_loss = val_loss

			f_scores.append(f1)
			val_losses.append(val_loss)

			if f_scores[-1] > highest_f:
				np.save(os.path.join(outputdir, "seq_clust_thresh_best_f1.npy"), np.array(thresh))
				torch.save(self.state_dict(), os.path.join(outputdir, "seq_clust_model_best_f1.pth"))
				highest_f = f_scores[-1]
		np.save(os.path.join(outputdir, "seq_clust_thresh.npy"), np.array(thresh))

		np.save(os.path.join(outputdir, "train_files.npy"), np.array(files))
		np.save(os.path.join(outputdir, "val_files.npy"), np.array(eval_files))
		torch.save(self.state_dict(), os.path.join(outputdir, "seq_clust_model.pth"))
		return loss_vals, val_losses, f_scores

	def evaluate(self, files, device, dir=None, model=None):
		if model is not None:
			self.load_state_dict(torch.load(os.path.join(model, "seq_clust_model.pth"), map_location=device))
		if dir is not None:
			files = [f for f in glob.glob(os.path.join(dir, "**/*.osu"), recursive=True)]
		ground = []
		loss_fn1 = nn.CrossEntropyLoss()
		loss_fn2 = nn.BCEWithLogitsLoss()
		running_loss = 0
		dataset_len = 0
		with torch.no_grad():
			self.eval()
			predictions = []
			combo_pred = []
			ground = []
			combo_ground = []

			for i, file in tqdm(enumerate(files)):
				try:
					dataset = TypeDataset(file)
					loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)
					dataset_len += len(loader)
					for i, (batch_X, batch_Y, batch_Z) in enumerate(loader):
						out1, out2, _, _ = self(batch_X.to(device))
						loss1 = loss_fn1(out1.view(-1, 3), batch_Y.to(device))
						loss2 = loss_fn2(out2.view(-1), batch_Z.to(device))
						loss = loss1 + loss2
						running_loss += loss.item()

						predictions.extend(torch.argmax(out1.cpu(), dim=1))
						ground.extend(batch_Y.numpy())
						combo_pred.extend(out2.cpu())
						combo_ground.extend(batch_Z.cpu())
				except FileNotFoundError as e:
					print(str(e))
					files.remove(file)

			predictions = np.array(predictions)
			ground = np.array(ground)
			combo_pred = np.array(combo_pred)
			combo_ground = np.array(combo_ground)

		print(combo_pred)
		pr, re, thresh =  precision_recall_curve(combo_ground, combo_pred)
		fscore = (2*pr*re)/(pr+re)
		ix = np.argmax(fscore)
		print("Best:", thresh[ix], "f1score:", fscore[ix])
		print(running_loss/dataset_len)
		sequence_f1 = f1_score(ground, predictions, average='micro')

		combo_threshed = np.zeros(len(combo_pred))
		for i, pred in enumerate(combo_pred):
			if pred >= thresh[ix]:
				combo_threshed[i] = 1
		print(combo_threshed)
		combo_f1 = f1_score(combo_ground, combo_threshed)
		print("ppl:", torch.exp(torch.tensor(running_loss/dataset_len)))
		print("seqf1:", sequence_f1)
		print("combof1:", combo_f1)
		print((sequence_f1 + combo_f1) / 2)
		return running_loss/dataset_len, ((sequence_f1 + combo_f1) / 2), thresh[ix], torch.exp(torch.tensor(running_loss/dataset_len))

	def infer(self, onsets, target_diff, sections, global_tempo, local_tempo, device, model="..\\models\\default"):
		self.load_state_dict(torch.load(os.path.join(model, "seq_clust_model.pth"), map_location=device))
		thresh = np.load(os.path.join(model, "seq_clust_thresh.npy"))
		predictions = []
		combo_preds = []
		with torch.no_grad():
			self.eval()
			h_0 = None
			c_0 = None
			prev_time = 0
			out = 0
			curr_tempo = -1
			tempo = (1 / local_tempo[0][1]) * 60 * 1000
			past_var_feat = deque([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, convert_time(onsets[0], (1 / local_tempo[0][1]) * 60 * 1000)]], maxlen=3)
			const_t = np.array(global_tempo)
			difficulty = target_diff
			for x in tqdm(range(onsets[:-1].shape[0])):
				for (t, flag, _, _) in np.flip(sections):
					if t < x:
						if flag == -1 and target_diff != 0:
							difficulty = target_diff - 1
						elif flag == 1 and target_diff != 5:
							difficulty = target_diff + 1
						else:
							difficulty = target_diff
				const_feat = np.append(const_t, np.eye(6)[difficulty])

				if curr_tempo + 1 < local_tempo.shape[0]:
					if onsets[x] >= local_tempo[curr_tempo][0]:
						curr_tempo += 1
						tempo = (1 / local_tempo[curr_tempo][1]) * 60 * 1000

				if out == 1:
					typ = np.eye(3)[out]
					out = 0
					predictions.append(2)
					combo_preds.append(0)
					prev_time = onsets[x] - prev_time
					next_time = onsets[x + 1] - onsets[x]
					past_var_feat.append(np.append(typ, [0, convert_time(prev_time, tempo), convert_time(next_time, tempo)]))
					continue
				if out == 2:
					typ = np.eye(3)[out]
					out = 0
					predictions.append(5)
					combo_preds.append(0)
					prev_time = onsets[x] - prev_time
					next_time = onsets[x + 1] - onsets[x]
					past_var_feat.append(np.append(typ, [0, convert_time(prev_time, tempo), convert_time(next_time, tempo)]))
					continue
				input = []
				features = list(past_var_feat)
				for i in features:
					frame = np.append(const_feat, i)
					input.append(frame)
				input = torch.from_numpy(np.array(input)).float()

				out, combo, h_0, c_0 = self(input.view(-1, 3, 13).to(device), h_0, c_0)
				out = torch.argmax(out.view(3), dim=0).cpu()
				
				if convert_time(onsets[x + 1] - onsets[x], tempo) > 2 and out == 1:
					out == 0

				combo = combo.cpu().item()
				if combo > thresh:
					combo = 1
				else:
					combo = 0
				combo_preds.append(combo)
				
				typ = np.eye(3)[out]
				if out == 2:
					predictions.append(4)
				else:
					predictions.append(out)

				prev_time = onsets[x] - prev_time
				next_time = onsets[x + 1] - onsets[x]
				
				past_var_feat.append(np.append(typ, [combo, convert_time(prev_time, tempo), convert_time(next_time, tempo)]))

			if out == 1:
				combo_preds.append(0)
			elif out == 2:
				combo_preds.append(0)
			else:
				input = []
				features = list(past_var_feat)
				for i in features:
					frame = np.append(const_feat, i)
					input.append(frame)
				input = torch.from_numpy(np.array(input)).float()
				out, combo, h_0, c_0 = self(input.view(-1, 3, 13).to(device), h_0, c_0)
				out = torch.argmax(out.view(3), dim=0).cpu()
				if out == 1 or out == 2:
					out = 0
				combo = combo.cpu().item()
				if combo > thresh:
					combo = 1
				else:
					combo = 0
				combo_preds.append(combo)
			predictions.append(out)
		return np.array(predictions), np.array(combo_preds)


def prob_func(combo_len):
	return -0.3038 + 3.3241 / combo_len 

def cluster_onsets(onsets, tempo):
	random.seed(onsets[0])
	n_combo = np.zeros_like(onsets)
	n_combo[0] = 1
	combo_len = 1
	prev_onset = onsets[0]
	local_avg = 0
	curr_tempo = -1
	for i, onset in enumerate(onsets[1:-1]):
		if curr_tempo + 1 < tempo.shape[0]:
			if onset >= tempo[curr_tempo + 1][0]:
				curr_tempo += 1
		dist = convert_time(onset - prev_onset, 1 / tempo[curr_tempo][1] * 60 * 1000)
		if n_combo[i] == 1:
			local_avg = dist
		else:
			local_avg += dist
			local_avg /= 2
		if dist > (local_avg + 0.1) or dist > 4.95:
			n_combo[i + 1] = 1
			combo_len = 0
		elif round(combo_len / 2) >= 4:
			if random.random() > prob_func(combo_len):
				n_combo[i + 1] = 1
				combo_len = 0
		combo_len += 1
		prev_onset = onset
	return n_combo
