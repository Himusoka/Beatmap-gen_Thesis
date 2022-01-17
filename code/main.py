import argparse
from multiprocessing import Process, Queue

from modules.structure_module.structure_module import get_structure
from modules.tempo_module import get_tempo
import numpy as np

def float_range(mini,maxi):
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError("must be in range [" + str(mini) + " .. " + str(maxi)+"]")
        return f

    # Return function handle to checking function
    return float_range_checker

def pipeline(file, difficulty, model_paths, algo, outpath, style):
	print("Starting music structure analysis")
	structure = get_structure(file)
	print(structure)
	print("Finished music structure analysis")

	print("Starting music tempo analysis")
	q = Queue()
	p = Process(target=get_tempo, args=(file, q))
	p.start()
	global_bpm = q.get()
	local_bpm = q.get()
	p.join()
	print(global_bpm)
	print(local_bpm)
	print("Finished music tempo analysis")


	import torch
	from modules.onset_module import CBlstm
	from modules.clustering_module import LstmClustering, cluster_onsets
	from modules.sequence_module import Lstm
	from modules.generation_module import GenerationModule

	device = torch.device("cpu")
	if torch.cuda.is_available():
		device = torch.device("cuda:0")
		print("cuda")
	else:
		print("cpu")

	print("Starting onset detection")
	onset_predictor = CBlstm().to(device)
	onsets = onset_predictor.infer(
		file, difficulty, structure[0],
		local_bpm, device, model=model_paths)
	print("Finished onset detection:", onsets.shape[0], "onsets found")

	if algo == "lstm":
		print("Starting sequence generation + clustering")
		seq_clus_predicter = LstmClustering().to(device)
		sequence, new_combos = seq_clus_predicter.infer(
			onsets, difficulty, structure[0], global_bpm,
			local_bpm, device, model=model_paths)
		print("Finished sequence generation + clustering:", np.count_nonzero(new_combos), "groups")
	else:
		print("Starting onset clustering")
		new_combos = cluster_onsets(onsets, local_bpm)
		print("Finished onset clustering:", np.count_nonzero(new_combos), "groups")
		print("Starting sequence generation")
		sequence_predictor = Lstm().to(device)
		sequence = sequence_predictor.infer(
			onsets, new_combos, difficulty, structure[0],
			global_bpm, local_bpm, device, model=model_paths)
		print("Finished sequence generation")

	print("Starting Beatmap generation")
	generator = GenerationModule()
	bm = generator.generate_beatmap(
		file,
		difficulty,
		structure,
		global_bpm, local_bpm,
		onsets,
		new_combos,
		sequence,
		style,
		outpath)
	print("Finished .osu beatmap generation")
	return bm, structure, global_bpm, local_bpm

def pipeline_given_sec(file, difficulty, model_paths, algo, outpath, style, structure, global_bpm, local_bpm):

	import torch
	from modules.onset_module import CBlstm
	from modules.clustering_module import LstmClustering, cluster_onsets
	from modules.sequence_module import Lstm
	from modules.generation_module import GenerationModule

	device = torch.device("cpu")
	if torch.cuda.is_available():
		device = torch.device("cuda:0")
		print("cuda")
	else:
		print("cpu")

	print("Starting onset detection")
	onset_predictor = CBlstm().to(device)
	onsets = onset_predictor.infer(
		file, difficulty, structure[0],
		local_bpm, device, model=model_paths)
	print("Finished onset detection:", onsets.shape[0], "onsets found")

	if algo == "lstm":
		print("Starting sequence generation + clustering")
		seq_clus_predicter = LstmClustering().to(device)
		sequence, new_combos = seq_clus_predicter.infer(
			onsets, difficulty, structure[0], global_bpm,
			local_bpm, device, model=model_paths)
		print("Finished sequence generation + clustering:", np.count_nonzero(new_combos), "groups")
	else:
		print("Starting onset clustering")
		new_combos = cluster_onsets(onsets, local_bpm)
		print("Finished onset clustering:", np.count_nonzero(new_combos), "groups")
		print("Starting sequence generation")
		sequence_predictor = Lstm().to(device)
		sequence = sequence_predictor.infer(
			onsets, new_combos, difficulty, structure[0],
			global_bpm, local_bpm, device, model=model_paths)
		print("Finished sequence generation")

	print("Starting Beatmap generation")
	generator = GenerationModule()
	bm = generator.generate_beatmap(
		file,
		difficulty,
		structure,
		global_bpm, local_bpm,
		onsets,
		new_combos,
		sequence,
		style,
		outpath)
	print("Finished .osu beatmap generation")
	return bm


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-s", "--style",
		type=float_range(0, 1), default=0.5,
		help="the desired mapping style; A float between [0, 1] with asthetic represented by 0 and flow by 1")
	parser.add_argument(
		"-d", "--difficulty",
		type=int, default=4, choices=[0, 1, 2, 3, 4, 5],
		help="the desired difficulty of the generated beatmap; 0 - Easy, 5 - Expert+")
	parser.add_argument(
		"-a", "--algo", choices=["lstm", "proximity"],
		default="proximity", help="the choosen clustering algorithm")
	parser.add_argument(
		"-m", "--model", default="..\\models\\skystar",
		help="the trained model utilized for inference")
	parser.add_argument(
		"file", help="path of the audio file to generate the osu beatmap")
	parser.add_argument(
		"-o", "--out", default=".\\",
		help="output path of the generated .osu file")
	args = parser.parse_args()
	file = args.file
	out = args.out
	model_paths = args.model
	algo = args.algo
	difficulty = args.difficulty
	style = args.style
	pipeline(file, difficulty, model_paths, algo, out, style)

	