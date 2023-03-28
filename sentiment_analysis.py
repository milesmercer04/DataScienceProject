import pandas as pd
# Read CSV file to a pandas dataframe
df1 = pd.read_csv("scripts/SW_EpisodeIV.txt", sep=' ', escapechar='\\')
df2 = pd.read_csv("scripts/SW_EpisodeV.txt", sep=' ', escapechar='\\')
df3 = pd.read_csv("scripts/SW_EpisodeVI.txt", sep=' ', escapechar='\\')
# print(df)

# lines = df1.dialogue
movies = [df1, df2, df3]

# Code taken from a friend's Hackathon project
# The training data was adapted to my purposes but this code remains mostly unmodified
# Source code available here:
# https://github.com/Macbee280/CrimsonCode2023/blob/main/translationLayer.py
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu.model import Interpreter
from rasa_nlu import config
import spacy
import pathlib

train_data = load_data('sentiments.json')
nlu_config = config.load('config_spacy.yml')
nlp = spacy.load('en_core_web_lg')
trainer = Trainer(nlu_config)
trainer.train(train_data)
model_directory = trainer.persist('./')
model_path = pathlib.Path(model_directory)
interpreter = Interpreter.load(model_path)

def translationLayer(inputText):
    result = interpreter.parse(inputText)
    intent = result['intent']['name']
    # entities = result['entities']    
    return intent

for movie in movies:
    intents = []
    intentBinary = []
    for line in movie.dialogue:
        intents.append(translationLayer(line))
    for val in intents:
        if val == 'positive':
            intentBinary.append(1)
        elif val == 'negative':
            intentBinary.append(0)
        else:
            print('INVALID INTENT')
            quit()
    movie.insert(3, 'sentiment', intentBinary)

df1.to_csv('scripts/EpisodeIV_Sentiments.csv', index=False)
df2.to_csv('scripts/EpisodeV_Sentiments.csv', index=False)
df3.to_csv('scripts/EpisodeVI_Sentiments.csv', index=False)

# Going forward:
#   - Verify that AI is producing reliable assessments or train it more
#   - Put sentiments into dataframe as a new column (I don't know how yet)
#   - Trace change in average sentiment over the course of a movie
#   - See which characters are the most positive/negative
#   - Try to automate cutting up scripts into separate dialogues to analyze sentiment in interactions