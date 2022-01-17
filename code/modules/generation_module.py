import random
import math

import numpy as np
from tqdm import tqdm

from utilities.osu_serializer import OsuSerializer

class MomentumFlow():
	patterns = [np.array([[-math.cos(math.radians(35)), math.sin(math.radians(35)), 1.5]]),
	np.array([[0, 1, 1], [-1, 0, 1]]),
	np.array([[-1, 0, 1],
			  [math.cos(math.radians(36)), -math.sin(math.radians(36)), 1],
			  [-math.cos(math.radians(72)), math.sin(math.radians(72)), 1]]),
	np.array([[1, 0, 0]]),
	np.array([[1, 0, 1.5]]),
	np.array([[0, 1, 0.6], [-1, 0, 1]]),
	np.array([[-math.cos(math.radians(45)), math.sin(math.radians(45)), 1/math.cos(math.radians(45))], [1, 0, 1]])]

	pattern_types = np.array([[0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 0, 0, 0],
		[1, 2, 0],
		[1, 2, 0],
		[1, 2, 1, 2],
		[1, 2, 1, 2]])

	# change to next momentum piece in four circle piece in coordination system
	def __init__(self, seed, distance, diff):
		random.seed(seed)
		self.last_point = np.array([int(random.random() * 512), int(random.random() * 384)])
		self.current_point = np.array([int(random.random() * 512), int(random.random() * 384)])
		self.distance = distance
		self.rotation = 0
		if diff <= 1:
			self.diff_mult = 0.5
		elif diff == 2:
			self.diff_mult = 0.65
		elif diff == 3:
			self.diff_mult = 0.8
		elif diff == 4:
			self.diff_mult = 0.95
		else:
			self.diff_mult = 1

	
	def _calc_dist(self, time, beat_length, dist_mult, slider_vel, start=False):
		time_dist = ((time/beat_length) * 4)
		if start:
			dist_mult *= 1.15
		slider_add = 133 / (1 + math.exp(-2.5 * slider_vel * dist_mult + 6))
		distance = (33 * self.diff_mult * time_dist) + slider_add
		return distance 

	def _calc_best_interval(self, dist, angle, pos):
		best_interval = []
		intervals = []
		interval_part = []
		for i in range(0, 51): # rotate by 0 - 150 deg
			rota = (i/60 * math.pi)
			x_dir = math.cos(angle + rota)
			y_dir = math.sin(angle + rota)
			x = int(pos[0] + dist * x_dir)
			y = int(pos[1] + dist * y_dir)

			if not(x > 512 or x < 0 or y > 384 or y < 0):
				interval_part.append(abs(((i - 1)/60 * math.pi)))
			else:
				if interval_part:
					intervals.append(interval_part)
					interval_part = []
		if interval_part:
			intervals.append(interval_part)
		if intervals:
			if len(intervals) > 1:
				biggest = 0
				for interval in intervals:
					if len(interval) >= biggest:
						best_interval = [min(interval), max(interval)]
						biggest = len(interval)
			else:
				best_interval = [min(intervals[0]), max(intervals[0])]
		return best_interval

	
	def put_random(self, next_len, beat_length, dist_mult, slider_vel, dist=None):
		self.last_point = self.current_point
		new_rot = random.random() * 2 * math.pi
		if dist is None:
			self.current_point = np.array([int(random.random() * 512), int(random.random() * 384)])
		else:
			pos = [-1, -1]
			while pos[0] > 512 or pos[0] < 0 or pos[1] > 384 or pos[1] < 0:
				new_rot = random.random() * 2 * math.pi
				x_dir = math.cos(new_rot)
				y_dir = math.sin(new_rot)
				self.current_point = np.array([int(self.last_point[0] + dist * x_dir), int(self.last_point[1] + dist * y_dir)])
		next_dist = self._calc_dist(next_len, beat_length, dist_mult, slider_vel)
		if next_len/beat_length >= 1.95:
			next_dist = 1
		interval = 0
		limiter = 6
		while interval < 75:
			limiter -= 1
			if limiter < 0:
				limiter = 6
				if dist is None:
					self.current_point = np.array([int(random.random() * 512), int(random.random() * 384)])
				else:
					pos = [-1, -1]
					while pos[0] > 512 or pos[0] < 0 or pos[1] > 384 or pos[1] < 0 and limiter >= 0:
						new_rot = random.random() * 2 * math.pi
						x_dir = math.cos(new_rot)
						y_dir = math.sin(new_rot)
						self.current_point = np.array([int(self.last_point[0] + dist * x_dir), int(self.last_point[1] + dist * y_dir)])
					if limiter < 0:
						dist = None
						self.current_point = np.array([int(random.random() * 512), int(random.random() * 384)])
					limiter = 6
			x_dir = math.cos(new_rot)
			y_dir = math.sin(new_rot)
			pos = [int(self.current_point[0] + next_dist * x_dir), int(self.current_point[1] + next_dist * y_dir)]
			if pos[0] > 512 or pos[0] < 0 or pos[1] > 384 or pos[1] < 0:
				new_rot = random.random() * 2 * math.pi
				continue

			best_interval = self._calc_best_interval(next_dist, new_rot, pos)
			if not best_interval:
				new_rot = random.random() * 2 * math.pi
				continue

			max_next_angle = best_interval[1]
			min_next_angle = best_interval[0]
			interval = abs(math.degrees(max_next_angle) - math.degrees(min_next_angle))
			if interval < 75:
				new_rot = random.random() * 2 * math.pi



		self.rotation = new_rot
		return self.current_point
	

	def get_next_point(self, curr_len, next_len, beat_length, dist_mult, slider_vel, dist=None, start=False):
		max_next_angle = 0
		interval_limit = 75

		if dist is not None:
			curr_dist = dist
		else:
			curr_dist = self._calc_dist(curr_len, beat_length, dist_mult, slider_vel, start=start)
		next_dist = self._calc_dist(next_len, beat_length, dist_mult, slider_vel, start=start)
		curr_limit = self._calc_best_interval(curr_dist, self.rotation, self.current_point)


		if (curr_len/beat_length) >= 1.70 or not curr_limit:
			#self.clock_wise = not self.clock_wise
			self.put_random(next_len, beat_length, dist_mult, slider_vel)
			return self.current_point
			
		if (curr_len/beat_length) <= 0.3 and dist is None:
			#limit rotation 0 to 60 deg
			if curr_limit[1] >= math.radians(60):
				curr_limit[1] = math.radians(60)
				interval_limit = 30
		
		if (next_len/beat_length) <= 1.70:
			next_limit_int = 0
			while next_limit_int < interval_limit:
				x_dir_up = math.cos(self.rotation + curr_limit[1])
				y_dir_up = math.sin(self.rotation + curr_limit[1])
				pos_up = [int(self.current_point[0] + curr_dist * x_dir_up), int(self.current_point[1] + curr_dist * y_dir_up)]
				
				best_next_int_up = self._calc_best_interval(next_dist, curr_limit[1], pos_up)

				if not best_next_int_up:
					curr_limit[1] -= 1/30 * math.pi
					if curr_limit[1] <= curr_limit[0]:
						self.put_random(next_len, beat_length, dist_mult, slider_vel)
						return self.current_point
					continue
				max_next_angle = best_next_int_up[1]
				min_next_angle = best_next_int_up[0]
				next_limit_int = abs(math.degrees(max_next_angle) - math.degrees(min_next_angle))
				if pos_up[0] > 512 or pos_up[0] < 0 or pos_up[1] > 384 or pos_up[1] < 0:
					next_limit_int = 0
				if next_limit_int < interval_limit:
					curr_limit[1] -= 1/30 * math.pi
					if curr_limit[1] <= curr_limit[0]:
						self.put_random(next_len, beat_length, dist_mult, slider_vel)
						return self.current_point

			next_limit_int = 0
			while next_limit_int < interval_limit:
				x_dir_down = math.cos(self.rotation + curr_limit[0])
				y_dir_down = math.sin(self.rotation + curr_limit[0])
				pos_down = [int(self.current_point[0] + curr_dist * x_dir_down), int(self.current_point[1] + curr_dist * y_dir_down)]
				best_next_int_down = self._calc_best_interval(next_dist, curr_limit[0], pos_up)
				if not best_next_int_down:
					curr_limit[0] += 1/30 * math.pi
					if curr_limit[0] >= curr_limit[1]:
						self.put_random(next_len, beat_length, dist_mult, slider_vel)
						return self.current_point
					continue

				max_next_angle = best_next_int_down[1]
				min_next_angle = best_next_int_down[0]
				next_limit_int = abs(math.degrees(max_next_angle) - math.degrees(min_next_angle))
				if pos_down[0] > 512 or pos_down[0] < 0 or pos_down[1] > 384 or pos_down[1] < 0:
					next_limit_int = 0
				if next_limit_int < interval_limit:
					curr_limit[0] += 1/30 * math.pi
					if curr_limit[0] >= curr_limit[1]:
						self.put_random(next_len, beat_length, dist_mult, slider_vel)
						return self.current_point

		curr_interval = abs(curr_limit[1] - curr_limit[0]) 
		rot_dir = 1 
		angle = self.rotation + (random.random() * curr_interval + curr_limit[0]) * rot_dir
		x_dir = math.cos(angle)
		y_dir = math.sin(angle)
		self.last_point = self.current_point
		self.current_point = np.array([int(self.current_point[0] + curr_dist * x_dir), int(self.current_point[1] + curr_dist * y_dir)])
		self.rotation = angle
		
		while self.rotation > (2*math.pi):
			self.rotation -= 2 * math.pi
		while self.rotation < -(2*math.pi):
			self.rotation += 2 * math.pi
		return self.current_point

	def put_pattern(self, pattern_lens, prev_len, pattern, beat_len, dist_mult, slider_vel):
		start_pos = self.get_next_point(pattern_lens[0] - prev_len, pattern_lens[1] - pattern_lens[0], beat_len, dist_mult, slider_vel, start=True)
		start_rotation = self.rotation
		rotation = math.degrees(start_rotation)
		start_last_point = self.last_point
		pattern_dirs = self.patterns[pattern]
		restart = True
		while restart:
			restart = False
			self.last_point = start_last_point
			self.current_point = start_pos
			self.rotation = start_rotation
			
			rotation += 5
			if abs(rotation - math.degrees(start_rotation)) >= 360:
				start_pos = self.put_random(pattern_lens[1] - pattern_lens[0], beat_len, dist_mult, slider_vel)
				rotation = math.degrees(self.rotation)
			tmp_points = [start_pos]
			self.get_next_point(pattern_lens[1] - pattern_lens[0], pattern_lens[2] - pattern_lens[1], beat_len, dist_mult, slider_vel)
			tmp_points.append(self.current_point)
	
			for i in range(2, len(pattern_lens)):
				x_dir = pattern_dirs[i - 2][0] + math.sin(self.rotation + math.radians(rotation))
				y_dir = pattern_dirs[i - 2][0] + math.cos(self.rotation + math.radians(rotation))
				dist = self._calc_dist(pattern_lens[i] - pattern_lens[i - 1], beat_len, dist_mult * pattern_dirs[i - 2][2], slider_vel)
				pos = np.array([int(self.current_point[0] + dist * x_dir), int(self.current_point[1] + dist * y_dir)])
				self.last_point = self.current_point
				self.current_point = pos
				tmp_points.append(pos)
				self.rotation = math.atan2(y_dir, x_dir)
				if pos[0] > 512 or pos[1] > 384 or pos[0] < 0 or pos[1] < 0:
					restart = True
					break
		return tmp_points
			

	def _calc_orthogonal(self, vector):
		denominator = math.sqrt(vector[0]**2 + vector[1]**2)
		x = - vector[1] / denominator
		y = vector[0] / denominator
		return np.array([x, y])

	def get_slider_details(self, n_points, cur_pos, prevpos):
		dist_vec = prevpos - cur_pos
		if math.sqrt(dist_vec[0]**2 + dist_vec[1]**2) < 70:
			n_points = 1
		if n_points == 1:
			return "L" + '|' + str(cur_pos[0]) + ':' + str(cur_pos[1])
		if n_points >= 2:
			orthogonal_vec = self._calc_orthogonal(dist_vec)
			dir = 25
			n_point = (prevpos - 0.5 * dist_vec) + dir * orthogonal_vec
			return "B|" + str(int(n_point[0])) + ':' + str(int(n_point[1])) + '|' + str(cur_pos[0]) + ':' + str(cur_pos[1])

	def get_current_slider_points(self, n_points, slider_len, next_len, beat_len, dist_mult, slider_vel):
		self.get_next_point(0, next_len, beat_len, dist_mult, slider_vel, dist=slider_len)
		dist_vec = self.last_point - self.current_point
		if math.sqrt(dist_vec[0]**2 + dist_vec[1]**2) < 70:
			n_points = 1
			return "L" + '|' + str(self.current_point[0]) + ':' + str(self.current_point[1])
		if n_points >= 2:
			dist_vec = self.last_point - self.current_point
			orthogonal_vec = self._calc_orthogonal(dist_vec)
			dir = 25
			n_point = (self.last_point - 0.5 * dist_vec) + dir * orthogonal_vec
			return "B|" + str(int(n_point[0])) + ':' + str(int(n_point[1])) + '|' + str(self.current_point[0]) + ':' + str(self.current_point[1])


class GenerationModule():
	a_rate = {
		0: [1, 5],
		1: [4, 6],
		2: [6, 8],
		3: [7, 9.3],
		4: [8, 10],
		5: [8, 10]}
	
	hp_drain = {
		0: [1, 3],
		1: [3, 5],
		2: [4, 6],
		3: [5, 8],
		4: [5, 10],
		5: [5, 10]
	}

	o_diff ={
		0: [1, 3],
		1: [3, 5],
		2: [5, 7],
		3: [7, 9],
		4: [8, 10],
		5: [8, 10]
	}

	def round_dur(self, dur):
		if dur <= 0.30:
			dur = 0.25
		elif dur <= 0.55:
			dur = 0.5
		elif dur <= 0.80:
			dur = 0.75
		elif dur <= 1.05:
			dur = 1.0
		else:
			dur = round(dur, 2)
		return dur


	def _is_pattern_in(self, pattern, pattern_list):
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

	def _shallow_is_pattern_in(self, pattern, pattern_list):
		matches = False
		for uniq_pattern in pattern_list:
			if len(pattern) == len(uniq_pattern):
				for idx in range(len(pattern)):
					if uniq_pattern[idx] != pattern[idx]:
						matches = False
						break
					else:
						matches = True
			if matches:
				break
		return matches

	def _get_index(self, pattern, pattern_list):
		matches = False
		for i, uniq_pattern in enumerate(pattern_list):
			if len(pattern) == len(uniq_pattern):
				for idx in range(len(pattern)):
					if uniq_pattern[idx] != pattern[idx]:
						matches = False
						break
					else:
						matches = True
			if matches:
				break
		if i == 3 or i == 5:
			if random.random() > 0.5:
				i += 1
		return i

	def convert_to_hit_objects(self, onsets, types, combos, global_bpm, timing_points, difficulty, style=1):
		if difficulty > 2:
			base_dist = round(random.random() + 1, 2)
		else:
			base_dist = round(random.random() * 0.45 + 0.8, 2)
		placement = MomentumFlow(onsets[0] * global_bpm, base_dist, difficulty)
		uniq_patterns = []
		group_n = []
		assigned = []
		aesthetic_pattern = 0
		flow_pattern = 0
		beat_len = timing_points[0]["beatLength"]

		pattern = [[self.round_dur((onsets[1] - onsets[0])/beat_len), types[0]]]
		for i in range(1, onsets.shape[0]):
			if combos[i] == 1 or i == onsets.shape[0] - 1:
				if self._is_pattern_in(pattern, uniq_patterns):
					uniq_patterns.append(pattern)
					group_n.append(uniq_patterns.index(pattern))
				else:
					group_n.append(len(uniq_patterns))

					uniq_patterns.append(pattern)
				pattern = []
				
			for timing_point in timing_points:
				if onsets[i] >= timing_point["time"] and timing_point["uninherited"] == 1:
					beat_len = timing_point["beatLength"]
			pattern.append([self.round_dur((onsets[i] - onsets[i - 1])/beat_len), types[i]])

		for pattern in uniq_patterns:
			pattern = np.array(pattern)
			pattern_type = pattern[:,1]
			pattern_dist = pattern[:,0]
			if self._shallow_is_pattern_in(pattern_type, placement.pattern_types) and pattern_dist.max() == pattern_dist.min():
				assigned.append(True)
			else:
				assigned.append(False)
		
		while 1 - (sum(assigned) / len(assigned)) < style:
			assigned[int(random.random() * len(assigned))] = False

		beat_len = timing_points[0]["beatLength"]
		if difficulty > 2:
			slider_vel = (1/beat_len * 60 * 1000)/((1/beat_len * 60 * 1000) + 120) + 0.5
		else:
			slider_vel = 1.25
		next_len = onsets[1] - onsets[0]
		pos = placement.put_random(next_len, beat_len, 1, slider_vel)
		if assigned[group_n[0]]:
			aesthetic_pattern += 1
		else:
			flow_pattern += 1

		i = 1
		group_idx = 1
		n_combo = 0
		pattern_idx = 0
		slider_rate = slider_vel
		dist_mult = base_dist
		hit_objects = []
		hits_pos = []
		if types[0] == 0:
			hit_objects.append({"x":pos[0], "y":pos[1], "time":int(onsets[0]), "type": 5, "hitSound":0, "objectParams":[], "hitSample": "0:0:0:0:"})
		if types[0] == 1:
			s_len = int((next_len * slider_rate * 100) / beat_len)
		while i < onsets.shape[0] - 1:
			for timing_point in timing_points:
				if onsets[i] >= timing_point["time"] and timing_point["uninherited"] == 0:
					if difficulty > 2:
						slider_rate = - 1 / (timing_point["beatLength"] / 100)
					else:
						slider_rate = slider_vel

				if onsets[i] >= timing_point["time"] and timing_point["uninherited"] == 1:
					beat_len = timing_point["beatLength"]
			prev_len = next_len
			next_len = onsets[i + 1] - onsets[i]

			if combos[i] == 1:
				n_combo = 4
				pattern_len = len(uniq_patterns[group_n[group_idx]])
				pattern_idx = 0
				hits_pos = []
				if assigned[group_n[group_idx]]:
					aesthetic_pattern += 1
					curr_pattern = np.array(uniq_patterns[group_n[group_idx]])[:,1]
					hits_pos = placement.put_pattern(onsets[i:i+ pattern_len], onsets[i - 1], self._get_index(curr_pattern, placement.pattern_types),beat_len, dist_mult, slider_vel)
				else:
					flow_pattern += 1
				group_idx += 1
			if types[i] == 1:
				if not (hits_pos and pattern_idx < len(hits_pos)):
					pos = placement.get_next_point(prev_len, next_len, beat_len, dist_mult, slider_rate, start=True if n_combo == 4 else False)
				s_len = int((next_len * slider_rate * 100) / beat_len)
			elif types[i] == 2:
				if hits_pos and pattern_idx < len(hits_pos):
					pos = hits_pos[pattern_idx]
					curve_details = placement.get_slider_details(2, pos, hits_pos[pattern_idx - 1])
					pattern_idx += 1
				else:
					curve_details = placement.get_current_slider_points(2, s_len, next_len, beat_len, 1, slider_rate)
				hit_objects.append({"x": pos[0], "y": pos[1],"time": int(onsets[i - 1]),"type": 2 + n_combo, "hitSound": 0, "objectParams":[curve_details, 1, s_len]})
				n_combo = 0
			elif types[i] == 4:
				pos = np.array([0,0])
			elif types[i] == 5:
				hit_objects.append({"x": pos[0], "y": pos[1], "time": int(onsets[i - 1]),"type": 4 + n_combo, "hitSound":0, "objectParams":[onsets[i]]})
				n_combo = 0
			else:
				if hits_pos and pattern_idx < len(hits_pos):
					pos = hits_pos[pattern_idx]
					pattern_idx += 1
				else:
					pos = placement.get_next_point(prev_len, next_len, beat_len, dist_mult, slider_rate, start=True if n_combo == 4 else False)
				hit_objects.append({"x":pos[0], "y":pos[1], "time":int(onsets[i]), "type": 1 + n_combo, "hitSound":0, "objectParams":[], "hitSample": "0:0:0:0:"})
				n_combo = 0
			i += 1
		if types[-1] == 2:
			if hits_pos and pattern_idx < len(hits_pos):
				pos = hits_pos[pattern_idx]
				curve_details = placement.get_slider_details(2, pos, hits_pos[pattern_idx - 1])
			else:
				curve_details = placement.get_current_slider_points(2, s_len, 0, beat_len, 1, slider_rate)
			hit_objects.append({"x": pos[0], "y": pos[1],"time": int(onsets[i - 1]),"type": 2 + n_combo, "hitSound": 0, "objectParams":[curve_details, 1, s_len]})
		elif types[-1] == 5:
			hit_objects.append({"x": pos[0], "y": pos[1], "time": int(onsets[i - 1]),"type": 4 + n_combo, "hitSound":0, "objectParams":[onsets[-1]]})
		else:
			if hits_pos and pattern_idx < len(hits_pos):
				pos = hits_pos[pattern_idx]
			else:
				pos = placement.get_next_point(prev_len, 0, beat_len, 1, slider_rate, start=True if n_combo == 4 else False)
			hit_objects.append({"x":pos[0], "y":pos[1], "time":int(onsets[-1]), "type": 1 + n_combo, "hitSound":0, "objectParams":[], "hitSample": "0:0:0:0:"})
		
		return hit_objects, flow_pattern, aesthetic_pattern
					


	def set_timing_points(self, onset, local_bpm, intensity, avg, difficulty):
		un_timing_points = [{"time":onset, "beatLength": 60/local_bpm[0][1]*1000,  "meter":4, "sampleSet":0, "sampleIndex":0, "volume":100, "uninherited":1, "effects":0}]
		for tempo in local_bpm[1:]:
			un_timing_points.append({"time":tempo[0],  "beatLength":60 / tempo[1] * 1000, "meter":4, "sampleSet":0, "sampleIndex":0, "volume":100, "uninherited":1, "effects":0})

		if intensity[0][1] and difficulty > 2:
			if (avg / intensity[0][2]) * -100 < -200:
				in_timing_points = [{"time": onset, "beatLength":-200, "meter":4, "sampleSet":0, "sampleIndex":0, "volume":100, "uninherited":0, "effects":0}]
			else:
				in_timing_points = [{"time": onset, "beatLength":(avg / intensity[0][2]) * -100, "meter":4, "sampleSet":0, "sampleIndex":0, "volume":100, "uninherited":0, "effects":0}]
		else:
			in_timing_points = [{"time": onset, "beatLength": -100, "meter":4, "sampleSet":0, "sampleIndex":0, "volume":100, "uninherited":0, "effects":0}]

		for t in intensity[1:]:
			if t[1] and difficulty > 2:
				if (avg / t[2]) * -100 < -200:
					slider_vel = -200
				else:
					slider_vel = (avg / t[2]) * -100	
			else:
				slider_vel = -100
			in_timing_points.append({"time": t[0], "beatLength": slider_vel, "meter":4, "sampleSet":0, "sampleIndex":0, "volume":100, "uninherited":0, "effects":t[3]})
		timing_points = [un_timing_points[0], in_timing_points[0]]

		i, j = 1, 1
		while i < len(un_timing_points) and j < len(in_timing_points):
			if un_timing_points[i]["time"] < in_timing_points[j]["time"]:
				timing_points.append(un_timing_points[i])
				i += 1
			else:
				timing_points.append(in_timing_points[j])
				j += 1
		if i >= len(un_timing_points):
			for in_timing_point in in_timing_points[j:]:
				timing_points.append(in_timing_point)
		else:
			for un_timing_point in un_timing_points[i:]:
				timing_points.append(un_timing_point)
		return timing_points

	def get_map_settings(self, difficulty):
		ar = round(random.random() * (self.a_rate[difficulty][1] - self.a_rate[difficulty][0]) + self.a_rate[difficulty][0])
		hp = round(random.random() * (self.hp_drain[difficulty][1] - self.hp_drain[difficulty][0]) + self.hp_drain[difficulty][0])
		od = round(random.random() * (self.o_diff[difficulty][1] - self.o_diff[difficulty][0]) + self.o_diff[difficulty][0])
		circle_size = round(random.random() + 4)
		return [ar, hp, od, circle_size]

	def generate_beatmap(self, file, difficulty, sections, global_bpm, local_bpm, onsets, combos, types, style, out):
		random.seed(onsets[0] * (difficulty + global_bpm))
		serializer = OsuSerializer()
			
		map_settings = self.get_map_settings(difficulty)
		timing_points = self.set_timing_points(onsets[0], local_bpm, sections[0], sections[1], difficulty)
		hit_objects, flow_pattern, aesthetic_pattern = self.convert_to_hit_objects(onsets, types, combos, global_bpm, timing_points, difficulty, style=style)
		return serializer.serialize(timing_points, hit_objects, map_settings, flow_pattern, aesthetic_pattern, file, difficulty, outpath=out)
