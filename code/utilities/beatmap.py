

class Beatmap():
	def __init__(self):
		self.path = ""
		self.version = 14
		self.general = {
			"AudioFilename" : "",
			"AudioLeadIn" : 0,
			"PreviewTime" : -1,
			"Countdown" : 0,
			"SampleSet" : "Normal",
			"StackLeniency" : 0.7,
			"Mode" : 0,
			#Bool
			#"LetterboxInBreaks" : 0,
			#Bool
			#"WidescreenStoryboard": 0,
			#            "extra" : {
			#Bool
			#                "UseSkinSprites" : 0,
			#                "OverlayPosition" : "",
			#                "SkinPreference" : "",
			#Bool
			#                "EpilepsyWarning" : 0,
			#               "CountdownOffset" : 0,
			#Bool
			#                "SpecialStyle": 0,
			#Bool
			#                "SamplesMatchPlaybackRate" : 0
			#            }
		}

		self.editor = {
			"Bookmarks" : [],
			"DistanceSpacing" : 1.1,
			"BeatDivisor" : 4,
			#"GridSize" : 4,
			#"TimelineZoom" : 1
		}
		self.metadata = {
			"Title" : "",#
			"TitleUnicode" : "",#
			"Artist" : "",#
			"ArtistUnicode" : "",#
			"Creator" : "",#
			"Version" : "",#
			"Source" : "",
			"Tags" : "",
			"BeatmapID" : 0,
			"BeatmapSetID" : -1
		}
		self.difficulty = {
			"HPDrainRate" : 5,
			"CircleSize" : 5,
			"OverallDifficulty" : 5,
			"ApproachRate" : 5,
			"SliderMultiplier" : 1.0,
			"SliderTickRate" : 1
		}
		self.events = []
		# time, beatLength, meter, sampleSet, sampleIndex, volume, uninherited, effects
		# int   double      int     int         int         int     bool        int
		self.timing_points = []
		self.colours = {}
		# x, y, time, type, hitSound, objectParams, hitSample
		#int int int  int   int         [,]          [:]
		self.hit_objects = []

		self.additional = {
			"length" : 0.0,
			"starRating" : 0,
			"nCircle" : 0,
			"nSlider" : 0,
			"nReverse" : 0,
			"nSpinner" : 0,
			"nObjects" : 0,
			"BPM" : 0,
			"aestheticPatterns" : 0,
			"flowPatterns" : 0
		}
