import os


from osu_sr_calculator import calculateStarRating

from utilities.beatmap import Beatmap


class OsuSerializer():
	difficulty_dict={
		0: "Easy",
		1: "Normal",
		2: "Hard",
		3: "Insane",
		4: "Expert",
		5: "Expert+"}


	def serialize_beatmap(self, bm):
		with open(bm.path, 'w', encoding= 'utf-8') as f:
			f.write("osu file format v" + str(bm.version) + '\n')

			f.write("\n[General]\n")
			for attribute in bm.general.keys():
				f.write(attribute + ": " + str(bm.general[attribute]) + '\n')

			f.write("\n[Editor]\n")
			for attribute in bm.editor.keys():
				if bm.editor[attribute]:
					f.write(attribute + ": " + (str(bm.editor[attribute])) + '\n')
				else:
					f.write(attribute + ": " + '\n')

			f.write("\n[Metadata]\n")
			for attribute in bm.metadata:
				f.write(attribute + ": " + str(bm.metadata[attribute]) + '\n')

			f.write("\n[Difficulty]\n")
			for attribute in bm.difficulty:
				f.write(attribute + ": " + str(bm.difficulty[attribute]) + '\n')

			f.write("\n[Events]\n")
			for attribute in bm.events:
				f.write(','.join(str(x) for x in attribute) + '\n')

			f.write("\n[TimingPoints]\n")
			for point in bm.timing_points:
				f.write(','.join(str(int(round(x)) if x != point["beatLength"] else x) for x in list(point.values())) + '\n')

			f.write("\n[Colours]\n")
			for attribute in bm.colours.keys():
				f.write(attribute + ": " + ','.join(str(x) for x in bm.colours[attribute]) + '\n')

			f.write("\n[HitObjects]\n")
			for hit in bm.hit_objects:
				hit_object = ""
				for elem in list(hit.values()):
					if isinstance(elem, list):
						if elem:
							hit_object += ','.join(str(x) for x in elem) + ','
					else:
						hit_object += str(elem) + ','
				f.write(hit_object[:-1] + '\n')
			f.close()

	def serialize(self, timing_points, hit_objects, map_settings, flow_pattern, aesthetic_pattern, audio, difficulty, outpath=".\\"):
		bm = Beatmap()
		bm.general["AudioFilename"] = audio.rsplit("\\", 1)[-1]
		bm.metadata["Title"] = audio.rsplit("\\")[-1].split(".")[0]
		bm.metadata["TitleUnicode"] = audio.rsplit("\\")[-1].split(".")[0]
		bm.metadata["Artist"] = "unknown"
		bm.metadata["ArtistUnicode"] = "unknown"
		bm.metadata["Creator"] = "MaGAI"
		bm.metadata["Version"] = self.difficulty_dict[difficulty]

		bm.difficulty["ApproachRate"] = map_settings[0]
		bm.difficulty["HPDrainRate"] = map_settings[1]
		bm.difficulty["OverallDifficulty"] = map_settings[2]
		bm.difficulty["CircleSize"] = map_settings[3]

		bm.timing_points = timing_points
		bm.hit_objects = hit_objects
		bm.additional["aestheticPatterns"] = aesthetic_pattern
		bm.additional["flowPatterns"] = flow_pattern
		if outpath is not None:
			bm.path = os.path.join(outpath, audio.split(".")[0] + "[" + self.difficulty_dict[difficulty] + "].osu")
			self.serialize_beatmap(bm)
			#print(calculateStarRating(filepath=bm.path)["nomod"])
		return bm
