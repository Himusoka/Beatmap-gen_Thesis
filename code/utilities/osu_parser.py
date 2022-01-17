from utilities.beatmap import Beatmap
from osu_sr_calculator import calculateStarRating

def isfloat(value):
	try:
		float(value)
		return True
	except ValueError:
		return False

def isint(value):
	try:
		int(value)
		return True
	except ValueError:
		return False

class OsuParser():
	def _parse_version(self, bm, attributes):
		bm.version = int(attributes[0].split('v')[1])

	def _parse_general(self, bm, attributes):
		attributes = [x.split(':') for x in attributes]
		for attribute in bm.general:
			if len(attributes) == 0:
				print("Missing attribute, using default value:")
				print(attribute + ':' + str(bm.general[attribute]))
			else:
				for x in attributes:
					if x[0] == attribute:
						if isint(x[1]):
							bm.general[x[0]] = int(x[1])
						elif isfloat(x[1]):
							bm.general[x[0]] = float(x[1])
						else:
							bm.general[x[0]] = str.strip(x[1])
						break
				attributes.remove(x)

	def _parse_editor(self, bm, attributes):
		attributes = [x.split(':') for x in attributes]
		for attribute in bm.editor:
			if len(attributes) == 0:
				print("Missing attribute, using default value:")
				print(attribute + ':' + str(bm.editor[attribute]))
			else:
				for x in attributes:
					if x[0] == attribute:
						if isint(x[1]):
							bm.editor[x[0]] = int(x[1])
						elif isfloat(x[1]):
							bm.editor[x[0]] = float(x[1])
						else:
							bm.editor[x[0]] = [int(y) for y in str.strip(x[1]).split(',')]
						break
				attributes.remove(x)

	def _parse_metadata(self, bm, attributes):
		attributes = [x.split(':') for x in attributes]
		for attribute in bm.metadata:
			if len(attributes) == 0:
				print("Missing attribute, using default value:")
				print(attribute + ':' + str(bm.metadata[attribute]))
			else:
				for x in attributes:
					if x[0] == attribute:
						if isint(x[1]):
							bm.metadata[x[0]] = int(x[1])
						else:
							if attribute == "Tags":
								bm.metadata[x[0]] = str.strip(x[1]).split(' ')
							else:
								bm.metadata[x[0]] = str.strip(x[1])
							break
				attributes.remove(x)

	def _parse_difficulty(self, bm, attributes):
		attributes = [x.split(':') for x in attributes]
		for attribute in bm.difficulty:
			if len(attributes) == 0:
				print("Missing attribute, using default value:")
				print(attribute + ':' + str(bm.difficulty[attribute]))
			else:
				for x in attributes:
					if x[0] == attribute:
						if x[0] in bm.difficulty:
							if isfloat(x[1]):
								bm.difficulty[x[0]] = float(x[1])
							break
				attributes.remove(x)

	def _parse_events(self, bm, attributes):
		return
		for x in attributes:
			eventhead = x.split(',')[:2]
			event_params = x.split(',')[2:]
			bm.events.append({
			"eventType" : eventhead[0],
			"startTime" : int(eventhead[1]),
			"eventParams" :event_params
		})

	def _parse_timing_points(self, bm, attributes):
		for x in attributes:
			x = x.split(',')
			bm.timing_points.append({
			"time" : int(float(x[0])),
			"beatLength" : float(x[1]),
			"meter" : int(x[2]),
			"sampleSet" : int(x[3]),
			"sampleIndex" : int(x[4]),
			"volume" : int(x[5]),
			"uninherited" : bool(int(x[6])),
			"effects" : int(x[7])
		})

	def _parse_colours(self, bm, attributes):
		attributes = [x.split(':') for x in attributes]
		for x in attributes:
			rgb = [int(y) for y in x[1].split(',')]
			bm.colours.update({x[0] : rgb})

	def _parse_slider(self, parameters):
		params = {}
		curves = parameters[0].split('|')
		params.update({"curveType" : curves[0]})
		points = []
		for x in curves[1:]:
			points.append(x.split(':'))
		params.update({"curvePoints" : points})
		params.update({"slides" : int(parameters[1])})
		params.update({"length" : float(parameters[2])})
		return params

	def _parse_hit_objects(self, bm, attributes):
		for x in attributes:
			x = x.split(',')
			if ':' not in x[-1]:
				x.append("0:0:0:0:")
			params = 0
			if int(x[3]) & 0x02 != 0:
				params = self._parse_slider(x[5:len(x) - 1])
			elif int(x[3]) & 0x08 != 0:
				params = int(x[5])
			bm.hit_objects.append({
			"x" : int(x[0]),
			"y" : int(x[1]),
			"time" : int(x[2]),
			"type" : int(x[3]),
			"hitSound" : int(x[4]),
			"objectParams" : params,
			"hitSample" : x[len(x) - 1]
		})

	def _parse_additional(self, bm):
		if bm.hit_objects:
			bm.additional["length"] = bm.hit_objects[-1]["time"]
			bm.additional["starRating"] = calculateStarRating(filepath=bm.path)["nomod"]
			for hit_object in bm.hit_objects:
				if hit_object["type"] & 1 == 1:
					bm.additional["nCircle"] += 1
				elif hit_object["type"] & 2 == 2:
					bm.additional["nSlider"] += 1
					if hit_object["objectParams"]["slides"] > 1:
						bm.additional["nReverse"] += hit_object["objectParams"]["slides"] - 1
				elif hit_object["type"] & 8 == 8:
					bm.additional["nSpinner"] += 1
			bm.additional["nObjects"] = len(bm.hit_objects)
		if bm.timing_points:
			relevant_points = [x for x in bm.timing_points if x["uninherited"]]
			tempo_len = {}
			for i in range(len(relevant_points)):
				if i + 1 >= len(relevant_points):
					length = bm.additional["length"] - relevant_points[i]["time"]
				else:
					length = relevant_points[i + 1]["time"] - relevant_points[i]["time"]
				if relevant_points[i]["beatLength"] in tempo_len:
					tempo_len[relevant_points[i]["beatLength"]] += length
				else:
					tempo_len[relevant_points[i]["beatLength"]] = length
			longest = 0
			for tempo in tempo_len:
				if tempo > longest:
					longest = tempo

			bm.additional["BPM"] = 1 / longest * 60 * 1000

	parse_sections = {
		"[General]" : _parse_general,
		"[Editor]" : _parse_editor,
		"[Metadata]" : _parse_metadata,
		"[Difficulty]" : _parse_difficulty,
		"[Events]" : _parse_events,
		"[TimingPoints]" : _parse_timing_points,
		"[Colours]" : _parse_colours,
		"[HitObjects]" : _parse_hit_objects
	}

	def parse_map(self, beatmapfile):
		bm = Beatmap()
		bm.path = beatmapfile
		print(beatmapfile)
		allines = []
		with open(beatmapfile, 'r', encoding="utf-8") as f:
			sectionlines = [str.rstrip(f.readline())]
			for line in f.readlines():
				if "//" in line:
					line = line.split("//")[0] + '\n'
				if '\n' == line:
					continue
				elif line.startswith('['):
					allines.append(sectionlines)
					sectionlines = [str.rstrip(line)]
				else:
					sectionlines.append(str.rstrip(line))
			allines.append(sectionlines)
		self._parse_version(bm, allines[0])
		for section in allines[1:]:
			self.parse_sections[section[0]](self, bm, section[1:])
		self._parse_additional(bm)
		return bm
