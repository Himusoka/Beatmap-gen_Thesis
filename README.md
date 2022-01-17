
# MaGAI
Implementation of an automatic beatmap creator for a thesis

# Requirements
Python 3.7

### Additional Requirements not in requirements.txt
FFMPEG
madmom
  
 # Usage
All scripts can be found in the code directory

#### Beatmap generation
In order to generate a beatmap the main.py script has to be executed with a mp3 file as input
Example:
```
python main.py -d 2 -m ..\\models\\skystar song_name.mp3
```
| Parameter | Argument | Description |
|------------|-------------------|-------------------------------------------------------------------------------------------------------|
| Difficulty | -d / --difficulty | The desired difficulty of the generated beatmap; 0 - Easy, 5 - Expert+ |
| Model | -m / --model | The trained model utilized for infering onsets and sequence generation |
| Style | -s / --style | A number between [0, 1] for the mapping style with aesthetic being 0 and circular flow 1\end{tabular} |
| Algorithm | -a / --algo | The chosen clustering algorithm, either "lstm" or "proximity" |
| Input | - | Path of the input audio file |
| Output | -o / --out | Path of the generated .osu file


#### Training
In order to train the modules the train_modules.py script has to be executed with the dataset path as input
Example:
```
python train_modules.py ..\\datasets\\skystar -c ..\\featureCache
```

| Parameter | Argument | Description |
|------------|-------------------|-------------------------------------------------------------------------------------------------------|
| Input | - | Path of the input dataset directory |
| Model | -c / --cache | Cache audio features for faster and future training |
| Output | -o / --out | Path of the trained models; default is the models directory


#### Evaluation
In order to evaluate a trained model the evaluate.py script has to be executed with the model path as input
Example:
```
python evaluate.py ..\\model\\skystar
```
Output can be found in eval directory

#### Metadata for Dataset
In order to extract metadata from a dataset the metadata_extractor.py script has to be executed with the dataset path as input
Example:
```
python metadata_extractor.py ..\\datasets\\skystar
```                                                                   |