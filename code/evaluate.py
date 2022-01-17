import os
import csv
import math
import argparse
from multiprocessing import Process, Queue


import numpy as np
from tqdm import tqdm
from osu_sr_calculator import calculateStarRating


from main import pipeline, pipeline_given_sec
from utilities.osu_parser import OsuParser
from utilities.feature_extractor import convert_diff, convert_time

def is_pattern_in(pattern, pattern_list):
		matches = False
		for uniq_pattern in pattern_list:
			if len(pattern) == len(uniq_pattern):
				for idx in range(len(pattern)):
					if uniq_pattern[idx][0] != pattern[idx][0] or uniq_pattern[idx][1] != pattern[idx][1]:
						matches = False
						break
					else:
						matches = True
			if matches:
				break
		return matches

def get_unique_patterns(bm):
	unique_patterns = []
	cur_pattern = []
	total_patterns = 0
	prev = bm.hit_objects[0]["time"]
	red_time_point = [y for y in bm.timing_points if y["uninherited"] == 1]
	cur_tempo_idx = 0
	for i in range(1, len(bm.hit_objects)):
		if cur_tempo_idx + 1 < len(red_time_point) and bm.hit_objects[i]["time"] >= red_time_point[cur_tempo_idx + 1]["time"]:
			cur_tempo_idx += 1

		cur_time = bm.hit_objects[i]["time"]
		cur_pattern.append([convert_time(cur_time - prev, red_time_point[cur_tempo_idx]["beatLength"]), bm.hit_objects[i]["type"]])
		if i + 1 < len(bm.hit_objects) and bm.hit_objects[i + 1]["type"] & 0x04 != 0 or i == len(bm.hit_objects) - 1:
			if not is_pattern_in(cur_pattern, unique_patterns):
				unique_patterns.append(cur_pattern)
			cur_pattern = []
			total_patterns += 1
		prev = cur_time
	return unique_patterns, total_patterns

def get_avg_pattern_len(unique_patterns):
	avg_pattern_len = 0
	for pattern in unique_patterns:
		avg_pattern_len += len(pattern)
	return avg_pattern_len / len(unique_patterns)

def get_difficulties_spread(audio, dataset_model):
	prox_diff_avg_delta = 0
	alt_diff_avg_delta = 0
	bm_prox = get_generated_bm(audio, 0, dataset_model, "proximity", ".\\tmp\\prox\\")
	bm_alt = get_generated_bm(audio, 0, dataset_model, "lstm", ".\\tmp\\alt\\")
	prev_prox_diff = calculateStarRating(filepath=bm_prox.path)["nomod"]
	prev_alt_diff = calculateStarRating(filepath=bm_alt.path)["nomod"]
	for diff in range(1, 6):
		bm_prox = get_generated_bm(audio, diff, dataset_model, "proximity", ".\\tmp\\prox\\")
		bm_alt = get_generated_bm(audio, diff, dataset_model, "lstm", ".\\tmp\\alt\\")
		curr_prox_diff = calculateStarRating(filepath=bm_prox.path)["nomod"]
		curr_alt_diff = calculateStarRating(filepath=bm_alt.path)["nomod"]
		prox_diff_avg_delta += curr_prox_diff - prev_prox_diff
		alt_diff_avg_delta += curr_alt_diff - prev_alt_diff
		os.remove(bm_prox.path)
		os.remove(bm_alt.path)
	return prox_diff_avg_delta / 6, alt_diff_avg_delta / 6

def get_distribution(hit_points):
	n_circle, n_slider, n_spinner = 0, 0, 0
	for hit in hit_points:
		if hit["type"] & 0x02 != 0:
			n_slider += 1
		elif hit["type"] & 0x08 != 0:
			n_spinner += 1
		else:
			n_circle += 1
	return [n_circle, n_slider, n_spinner]


def get_section_density_intensity(bm, sections):
	sections_avg_density = []
	sections_avg_velocity = []
	curr_sections_avg_density = 0
	curr_sections_avg_velocity = 0
	curr_sections_hits = 1
	prev = bm.hit_objects[0]["time"]
	prev_pos = np.array([bm.hit_objects[0]["x"], bm.hit_objects[0]["y"]])
	red_time_point = [y for y in bm.timing_points if y["uninherited"] == 1]
	cur_tempo_idx = 0
	cur_section = 0
	current = 0

	for hit in bm.hit_objects[1:]:
		if cur_section + 1 < len(sections) and hit["time"] >= sections[cur_section + 1]["time"]:
			sections_avg_density.append(curr_sections_avg_density / curr_sections_hits)
			sections_avg_velocity.append(curr_sections_avg_velocity / curr_sections_hits)
			curr_sections_hits = 0
			curr_sections_avg_density = 0
			curr_sections_avg_velocity = 0
			cur_section += 1
		if cur_tempo_idx + 1 < len(red_time_point) and hit["time"] >= red_time_point[cur_tempo_idx + 1]["time"]:
			cur_tempo_idx += 1

		current = hit["time"] - prev
		current_pos = np.array([hit["x"], hit["y"]])
		curr_sections_avg_density += convert_time(current, red_time_point[cur_tempo_idx]["beatLength"])

		distance = current_pos - prev_pos
		curr_sections_avg_velocity += math.sqrt(distance[0]**2 + distance[1]**2) / current

		curr_sections_hits += 1
		prev_pos = current_pos
		prev = hit["time"]
	sections_avg_density.append(curr_sections_avg_density / curr_sections_hits)
	sections_avg_velocity.append(curr_sections_avg_velocity / curr_sections_hits)
	cur_section += 1
	while cur_section + 1 < len(sections_avg_density):
		sections_avg_density.append(0)
		sections_avg_velocity.append(0)
		cur_section += 1

	return sections_avg_density, sections_avg_velocity

def get_avg_density(bm):
	avg_density = 0
	prev = bm.hit_objects[0]["time"]
	red_time_point = [y for y in bm.timing_points if y["uninherited"] == 1]
	cur_tempo_idx = 0
	for hit in bm.hit_objects[1:]:
		if cur_tempo_idx + 1 < len(red_time_point) and hit["time"] >= red_time_point[cur_tempo_idx + 1]["time"]:
			cur_tempo_idx += 1
		current = hit["time"] - prev
		avg_density += convert_time(current, red_time_point[cur_tempo_idx]["beatLength"])
		prev = hit["time"]

	return avg_density / (len(bm.hit_objects) - 1)

def get_generated_bm_func(audio_path, target_diff, model, algo, tmp, queue):
	bm, struc, g_temp, l_temp = pipeline(audio_path, target_diff, model, algo, tmp, 0.5)
	queue.put(bm)
	queue.put(struc)
	queue.put(g_temp)
	queue.put(l_temp)

def get_generated_bm_given_struc_func(audio_path, target_diff, model, algo, tmp, struc, g_temp, l_temp, queue):
	bm = pipeline_given_sec(audio_path, target_diff, model, algo, tmp, 0.5, struc, g_temp, l_temp)
	queue.put(bm)

def get_generated_bm_struc(audio_path, target_diff, model, algo, tmp, struc, g_temp, l_temp):
	q = Queue()
	p = Process(target=get_generated_bm_given_struc_func, args=(audio_path, target_diff, model, algo, tmp, struc, g_temp, l_temp, q))
	p.start()
	bm = q.get()
	p.join()
	return bm

def get_generated_bm(audio_path, target_diff, model, algo, tmp):
	q = Queue()
	p = Process(target=get_generated_bm_func, args=(audio_path, target_diff, model, algo, tmp, q))
	p.start()
	bm = q.get()
	struc = q.get()
	g_temp = q.get()
	l_temp = q.get()
	p.join()
	return bm, struc, g_temp, l_temp

def evaluate_seperate(dataset_model):

	osu_parser = OsuParser()
	val_files = np.load(os.path.join(dataset_model, "val_files.npy"))
	prev_audio =""

	total_objects, density, velocity, difficulties, dist_alt = [], [], [], [], []
	avg_dens, uniq_patterns, pattern_len, dist_prox = [], [], [], []
	style = []

	for file in tqdm(val_files):
		val_bm = osu_parser.parse_map(file)
		target_diff = convert_diff(val_bm.additional["starRating"])
		audio_path = os.path.join(val_bm.path.rsplit('\\', 1)[0], val_bm.general["AudioFilename"])
		if prev_audio == audio_path:
			bm_prox = get_generated_bm_struc(audio_path, target_diff, dataset_model, "proximity", ".\\tmp\\prox", struc, g_temp, l_temp)
		else:
			bm_prox, struc, g_temp, l_temp = get_generated_bm(audio_path, target_diff, dataset_model, "proximity", ".\\tmp\\prox")
		bm_alt = get_generated_bm_struc(audio_path, target_diff, dataset_model, "lstm", ".\\tmp\\alt", struc, g_temp, l_temp)
		
		sections = [y for y in bm_prox.timing_points if y["uninherited"] == 0]

		#difficulty
		try:
			bm_prox_diff = calculateStarRating(filepath=bm_prox.path)["nomod"]
			bm_alt_diff = calculateStarRating(filepath=bm_alt.path)["nomod"]
			#total_objects
			val_total = val_bm.additional["nObjects"]
			prox_total = len(bm_prox.hit_objects)
			alt_total = len(bm_alt.hit_objects)

			#density in sections
			val_density, val_velocity = get_section_density_intensity(val_bm, sections)
			prox_density, prox_velocity = get_section_density_intensity(bm_prox, sections)
			alt_density, alt_velocity = get_section_density_intensity(bm_alt, sections)
			if not(len(val_density) == len(prox_density) and len(prox_density) == len(alt_density)):
				lens = [len(val_density), len(prox_density), len(alt_density)]
				val_density = val_density[:min(lens)]
				prox_density = prox_density[:min(lens)]
				alt_density = alt_density[:min(lens)]
				val_velocity = val_velocity[:min(lens)]
				prox_velocity = prox_velocity[:min(lens)]
				alt_velocity = alt_velocity[:min(lens)]

			#density overall
			val_overall_dens = get_avg_density(val_bm)
			prox_overall_dens = get_avg_density(bm_prox)
			alt_overall_dens = get_avg_density(bm_alt)


			#prox_diff_spread, alt_diff_spread = get_difficulties_spread(audio_path, dataset_model)
			#diff_spread.append[prox_diff_spread, alt_diff_spread]

			#unique pattern
			val_uniq_patterns, val_total_patterns = get_unique_patterns(val_bm)
			prox_uniq_patterns, prox_total_patterns = get_unique_patterns(bm_prox)
			alt_uniq_patterns, alt_total_patterns = get_unique_patterns(bm_alt)


			#average pattern length
			val_avg_pattern_len = get_avg_pattern_len(val_uniq_patterns)
			prox_avg_pattern_len = get_avg_pattern_len(prox_uniq_patterns)
			alt_avg_pattern_len = get_avg_pattern_len(alt_uniq_patterns)

			#aesthetic vs flow
			prox_flow = bm_prox.additional["flowPatterns"]
			prox_aesth = bm_prox.additional["aestheticPatterns"]
			prox_style = prox_flow / (prox_flow + prox_aesth)

			alt_flow = bm_alt.additional["flowPatterns"]
			alt_aesth = bm_alt.additional["aestheticPatterns"]
			alt_style = alt_flow / (alt_flow + alt_aesth)

			difficulties.append([target_diff, convert_diff(bm_prox_diff), convert_diff(bm_alt_diff)])
			total_objects.append([val_total, prox_total, alt_total])
			density.append([val_density, prox_density, alt_density])

			velocity.append([val_velocity, prox_velocity, alt_velocity])
			avg_dens.append([val_overall_dens, prox_overall_dens, alt_overall_dens])
			uniq_patterns.append([
				(val_total_patterns / len(val_uniq_patterns)) / val_total_patterns, 
				(prox_total_patterns / len(prox_uniq_patterns)) / prox_total_patterns, 
				(alt_total_patterns /len(alt_uniq_patterns)) / alt_total_patterns])
			#distribution
			dist_prox.append(get_distribution(bm_prox.hit_objects))
			dist_alt.append(get_distribution(bm_alt.hit_objects))
			pattern_len.append([val_avg_pattern_len, prox_avg_pattern_len, alt_avg_pattern_len])
			style.append([0.5, prox_style, alt_style])
		except:
			print("pass")

		prev_audio = audio_path

	import torch
	from modules.onset_module import CBlstm
	from modules.clustering_module import LstmClustering
	from modules.sequence_module import Lstm
	device = torch.device("cpu")
	if torch.cuda.is_available():
		device = torch.device("cuda:0")
		print("cuda")

	#onset f1
	onset_predictor = CBlstm().to(device)
	_, onset_score, _= onset_predictor.evaluate(val_files, device, model=dataset_model)
	print(onset_score)

	#sequence f1
	seq_predictor = Lstm().to(device)
	_, seq_score, perplex_prox = seq_predictor.evaluate(val_files, device, model=dataset_model)
	print(seq_score)

	seq_clust_predictor = LstmClustering().to(device)
	_, seq_clust_score, _, perplex_alt = seq_clust_predictor.evaluate(val_files, device, model=dataset_model)
	print(seq_clust_score)

	outputdir = os.path.join("..\\eval", dataset_model.split('\\')[-1])
	if not os.path.exists(outputdir):
		os.mkdir(outputdir)

	with open(os.path.join(outputdir, 'f1.csv'), 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["onset", "sequence", "seq_clust", "seq_ppl", "seq_clust_ppl"])
		writer.writerow([onset_score, seq_score, seq_clust_score, perplex_prox, perplex_alt])

	with open(os.path.join(outputdir, 'objects.csv'), 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["validation", "proximity", "alternative"])
		writer.writerows(total_objects)

	with open(os.path.join(outputdir, 'difficulties.csv'), 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["target", "proximity", "alternative"])
		writer.writerows(difficulties)

	with open(os.path.join(outputdir, 'dist_alt.csv'), 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["circle", "slider", "spinner"])
		writer.writerows(dist_alt)

	with open(os.path.join(outputdir, 'dist_prox.csv'), 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["circle", "slider", "spinner"])
		writer.writerows(dist_prox)

	with open(os.path.join(outputdir, 'uniq_pattern.csv'), 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["validation", "proximity", "alternative"])
		writer.writerows(uniq_patterns)

	with open(os.path.join(outputdir, 'pattern_len.csv'), 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["validation", "proximity", "alternative"])
		writer.writerows(pattern_len)

	with open(os.path.join(outputdir, 'style.csv'), 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["target", "proximity", "alternative"])
		writer.writerows(style)

	with open(os.path.join(outputdir, 'overall_density.csv'), 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["val", "proximity", "alternative"])
		writer.writerows(avg_dens)

	with open(os.path.join(outputdir, 'density.csv'), 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["validation", "proximity", "alternative"])
		for bm_densities in density:
			writer.writerow(["nextbm"])
			for i in range(len(bm_densities[0])):
				writer.writerow([bm_densities[0][i], bm_densities[1][i], bm_densities[2][i]])

	with open(os.path.join(outputdir, 'velocity.csv'), 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["validation", "proximity", "alternative"])
		for bm in velocity:
			writer.writerow(["nextbm"])
			for i in range(len(bm[0])):
				writer.writerow([bm[0][i], bm[1][i], bm[2][i]])
	print("done")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("f", help="the model path")

	args = parser.parse_args()
	evaluate_seperate(args.f)
