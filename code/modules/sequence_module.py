from collections import deque
import os

import glob
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

from utilities.feature_extractor import FeatureExtractor, convert_time


BATCH_SIZE = 16
VAL_SIZE = 0.15
EPOCHS = 50
PATIENCE = 5
LR_RATE = 0.0002

class TypeDataset(Dataset):
	def __init__(self, file):
		self.extractor = FeatureExtractor()
		ground, data, _ = self.extractor.extract_types(file)
		self.x = torch.from_numpy(np.array(data))
		self.y = torch.from_numpy(np.array(ground))
		self.samples = self.x.shape[0]

	def __getitem__(self, index):
		return self.x[index].float(), self.y[index].long()

	def __len__(self):
		return self.samples

class Lstm(nn.Module):
	def __init__(self):
		super().__init__()
		self.lstm1 = nn.LSTM(input_size=13, hidden_size=128, batch_first=True, num_layers=2)
		self.lin = nn.Linear(3*128, 256)
		self.lin2 = nn.Linear(256, 128)
		self.out = nn.Linear(128, 3)
		self.soft = nn.Softmax(dim=1)

	def forward(self, x, h_t=None, c_t=None):
		if h_t is None or c_t is None:
			x, (h_n, c_n) = self.lstm1(x)
		else:
			x, (h_n, c_n) = self.lstm1(x, (h_t, c_t))
		x = F.relu(x)

		x = F.dropout(F.relu(self.lin(torch.flatten(x, start_dim=1))), training=self.training)
		x = self.out(F.dropout(F.relu(self.lin2(x)), training=self.training))
		if not self.training:
			x = self.soft(x)
		return x, h_n, c_n

	def start_training(self, dir, device, outputdir="..\\models\\default", ev_set=None, file_set=None):
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
		if ev_set is not None and file_set is not None:
			eval_files = np.load(ev_set)
			files = np.load(file_set)
		optimizer = optim.Adam(self.parameters(), lr=LR_RATE)
		loss_fn = nn.CrossEntropyLoss()
		loss_vals = []
		val_losses = []
		highest_f = 0
		loss_vals = []
		f_scores = []
		prev_val_loss = float('inf')
		training_patience = PATIENCE
		prev_state = self.state_dict()

		for epoch in range(EPOCHS):
			self.train()
			running_loss = 0
			datasetlen = 0
			np.random.shuffle(files)
			for i, file in enumerate(files):
				try:
					dataset = TypeDataset(file)
					loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)
					datasetlen += len(loader)
					print("Epoch: " + str(epoch) + "/" + str(EPOCHS) + ", data: " + str(i) + "/" + str(len(files)))
					for (batch_X, batch_Y) in tqdm(loader):
						optimizer.zero_grad()
						out, _, _ = self(batch_X.to(device))
						loss = loss_fn(out.view(-1, 3), batch_Y.to(device))
						loss.backward()
						optimizer.step()
						running_loss += loss.item()
				except FileNotFoundError as e:
					print(str(e))
					files.remove(file)
			train_loss = running_loss/datasetlen
			print("loss: ", train_loss)
			loss_vals.append(train_loss)
			val_loss, f1, _ = self.evaluate(eval_files, device)
			if prev_val_loss < val_loss:
				print("loss increased", abs(training_patience - 5))
				training_patience -= 1
				if training_patience == -1:
					print("Early training stop checkpoint after", epoch, "epochs")
					torch.save(prev_state, os.path.join(outputdir, "seq_model_check.pth"))
			else:
				prev_state = self.state_dict()
				training_patience = PATIENCE
			prev_val_loss = val_loss
			f_scores.append(f1)
			val_losses.append(val_loss)
			if f_scores[-1] > highest_f:
				torch.save(self.state_dict(), os.path.join(outputdir, "seq_model_best_f1.pth"))
				highest_f = f_scores[-1]

		np.save(os.path.join(outputdir, "train_files.npy"), np.array(files))
		np.save(os.path.join(outputdir, "val_files.npy"), np.array(eval_files))
		torch.save(self.state_dict(), os.path.join(outputdir, "seq_model.pth"))
		return loss_vals, val_losses, f_scores

	def evaluate(self, files, device, dir=None, model=None):
		if model is not None:
			self.load_state_dict(torch.load(os.path.join(model, "seq_model.pth"), map_location=device))
		if dir is not None:
			files = [f for f in glob.glob(os.path.join(dir, "**/*.osu"), recursive=True)]
		ground = []
		loss_fn = nn.CrossEntropyLoss()
		running_loss = 0
		dataset_len = 0
		with torch.no_grad():
			self.eval()
			predictions = []
			ground = []
			for i, file in tqdm(enumerate(files)):
				try:
					dataset = TypeDataset(file)
					loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)
					dataset_len += len(loader)
					for i, (batch_X, batch_Y) in enumerate(loader):
						out, _, _ = self(batch_X.to(device))
						loss = loss_fn(out.view(-1, 3), batch_Y.to(device))
						running_loss += loss.item()
						predictions.extend(torch.argmax(out.cpu(), dim=1))
						ground.extend(batch_Y.numpy())
				except FileNotFoundError as e:
					print(str(e))
					files.remove(file)
			print(out)
			predictions = np.array(predictions)
			ground = np.array(ground)

		print(running_loss/dataset_len)
		print("ppl:", torch.exp(torch.tensor(running_loss/dataset_len)))
		f1 = f1_score(ground, predictions, average='micro')
		print(f1)
		return running_loss/dataset_len, f1, torch.exp(torch.tensor(running_loss/dataset_len))

	def infer(self, onsets, combos, target_diff, sections, global_tempo, local_tempo, device, model="..\\models\\default"):
		self.load_state_dict(torch.load(os.path.join(model, "seq_model.pth"), map_location=device))
		predictions = []
		curr_tempo = -1
		with torch.no_grad():
			self.eval()
			h_0 = None
			c_0 = None
			prev_time = 0
			out = 0
			diff = target_diff
			past_var_feat = deque(
				[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 0, convert_time(onsets[0], (1 / local_tempo[0][1]) * 60 * 1000)]],
				maxlen=3)
			tempo = (1 / local_tempo[curr_tempo][1]) * 60 * 1000
			for x in tqdm(range(onsets[:-1].shape[0])):
				for (t, flag, _, _) in np.flip(sections):
					if t < x:
						if flag == -1 and target_diff != 0:
							diff = target_diff - 1
						elif flag == 1 and target_diff != 5:
							diff = target_diff + 1
						else:
							diff = target_diff
				const_f = np.append(tempo, np.eye(6)[diff])

				if curr_tempo + 1 < local_tempo.shape[0]:
					if onsets[x] >= local_tempo[curr_tempo + 1][0]:
						curr_tempo += 1
						tempo = (1 / local_tempo[curr_tempo][1]) * 60 * 1000

				if out == 1:
					typ = np.eye(3)[out]
					out = 0
					predictions.append(2)
					prev_time = onsets[x] - prev_time
					next_time = onsets[x + 1] - onsets[x]
					past_var_feat.append(
						np.append(
							typ,
							[0, convert_time(prev_time, tempo), convert_time(next_time, tempo)]))
					continue
				if out == 2:
					typ = np.eye(3)[out]
					out = 0
					predictions.append(5)
					prev_time = onsets[x] - prev_time
					next_time = onsets[x + 1] - onsets[x]
					past_var_feat.append(np.append(typ, [0, convert_time(prev_time, tempo), convert_time(next_time, tempo)]))
					continue

				input = []
				features = list(past_var_feat)
				for i in features:
					frame = np.append(const_f, i)
					input.append(frame)

				input = torch.from_numpy(np.array(input)).float()
				out, h_0, c_0 = self(input.view(-1, 3, 13).to(device), h_0, c_0)
				out = torch.argmax(out.view(3), dim=0).cpu()
				if convert_time(onsets[x + 1] - onsets[x], tempo) > 2 and out == 1:
					out == 0

				if combos[x + 1] == 1 and (out == 1 or out == 2):
					out = 0

				if out == 2:
					predictions.append(4)
				else:
					predictions.append(out)

				typ = np.eye(3)[out]
				prev_time = onsets[x] - prev_time
				next_time = onsets[x + 1]- onsets[x]
				past_var_feat.append(np.append(typ, [combos[x+1], convert_time(prev_time, tempo), convert_time(next_time, tempo)]))

			if out == 1:
				out = 2
			elif out == 2:
				out = 5
			else:
				input = []
				features = list(past_var_feat)
				for i in features:
					frame = np.append(const_f, i)
					input.append(frame)
				input = torch.from_numpy(np.array(input)).float()
				out, h_0, c_0 = self(input.view(-1, 3, 13).to(device), h_0, c_0)
				out = torch.argmax(out.view(3), dim=0).cpu()
				if out == 1 or out == 2:
					out = 0

			predictions.append(out)
		return np.array(predictions)
