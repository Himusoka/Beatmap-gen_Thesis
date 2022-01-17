import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from modules.onset_module import CBlstm
from modules.clustering_module import LstmClustering
from modules.sequence_module import Lstm

def show_train_plots(train_loss, val_loss, f1_score, module):
	plt.subplot(1, 2, 1)
	plt.plot(train_loss, '-o')
	plt.plot(val_loss, '-o')
	plt.xlabel("epoch")
	plt.ylabel("losses")

	plt.subplot(1, 2, 2)
	plt.plot(f1_score)
	plt.xlabel("epoch")
	plt.ylabel("F-score")
	plt.tight_layout()
	plt.draw()
	plt.savefig(module)
	plt.clf()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("p", help="the dataset path for the modules to train on")
	parser.add_argument("-c", "--cache", help="save audio features in specified directory")
	parser.add_argument("-o", "--output", default="..\\models\\")
	args = parser.parse_args()
	dataset = args.p
	cache = args.cache
	outpath = os.path.join(args.output, dataset.rsplit('\\', 1)[-1])

	device = torch.device("cpu")
	if torch.cuda.is_available():
		device = torch.device("cuda:0")
		print("cuda")

	onset_predictor = CBlstm().to(device)
	onset_train_l, onset_val_l, onset_f1 = onset_predictor.start_training(dataset, device, outputdir=outpath, cache=cache)
	show_train_plots(onset_train_l, onset_val_l, onset_f1, os.path.join(outpath, "onset_fig.png"))

	eval_files = os.path.join(outpath, "val_files.npy")
	train_files = os.path.join(outpath, "train_files.npy")

	sequence_predictor = Lstm().to(device)
	seq_train_l, seq_val_l, seq_f1 = sequence_predictor.start_training(dataset, device, outputdir=outpath, ev_set=eval_files, file_set=train_files)
	show_train_plots(seq_train_l, seq_val_l, seq_f1, os.path.join(outpath, "seq_fig.png"))

	seq_clus_predictor = LstmClustering().to(device)
	seq_clus_train_l, seqclus__val_l, seqclus__f1 = seq_clus_predictor.start_training(dataset, device, outputdir=outpath, ev_set=eval_files, file_set=train_files)
	show_train_plots(seq_clus_train_l, seqclus__val_l, seqclus__f1, os.path.join(outpath, "seq_clus_fig.png"))
