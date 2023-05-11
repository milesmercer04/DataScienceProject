# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import r_regression
from sklearn.preprocessing import PolynomialFeatures
from statistics import mean
from utils import *

# %%
# This cell reads the CSV's for the three movies into separate dataframes and makes an additional
# dataframe for all three movies combined
ep1 = pd.read_csv('scripts/EpisodeIV_Sentiments.csv')
ep2 = pd.read_csv('scripts/EpisodeV_Sentiments.csv')
ep3 = pd.read_csv('scripts/EpisodeVI_Sentiments.csv')

movies = [ep1, ep2, ep3]

full_script = pd.concat(movies)

# %%
# This cell separates the X and Y variables for regression and fits a regression model to the first movie
X1 = ep1[['line']].values.reshape(-1, 1)
y1 = ep1[['sentiment']].values.reshape(-1, 1)

polyModel = fitRegression(X1, y1, deg=3)

# %%
# This cell plots the regression for the first movie
plotRegression(X1, y1, polyModel, deg=3)
plt.xlabel('Line Number', fontsize=14)
plt.ylabel('Adjusted Sentiment Value', fontsize=14)
plt.title('Episode IV', loc='center')
plt.show()

# %%
# This cell prints the first movie's regression equation
print("Predicted Avg Sentiment =", polyModel.intercept_[0], "+",
      polyModel.coef_[0][0], "* (Line #) +", polyModel.coef_[0][1], "* (Line #)^2 +",
      polyModel.coef_[0][2], "* (Line #)^3")

# %%
# This cell fits a regression model to the data from the second movie
X2 = ep2[['line']].values.reshape(-1, 1)
y2 = ep2[['sentiment']].values.reshape(-1, 1)

polyModel = fitRegression(X2, y2, deg=2)

# %%
# This cell plots the regression for the second movie
plotRegression(X2, y2, polyModel, deg=2)
plt.xlabel('Line Number', fontsize=14)
plt.ylabel('Adjusted Sentiment Value', fontsize=14)
plt.title('Episode V', loc='center')
plt.show()

# %%
# This cell prints the second movie's regression equation
print("Predicted Avg Sentiment =", polyModel.intercept_[0], "+",
      polyModel.coef_[0][0], "* (Line #) +", polyModel.coef_[0][1], "* (Line #)^2")

# %%
# This cell fits a regression model to the third movie
X3 = ep3[['line']].values.reshape(-1, 1)
y3 = ep3[['sentiment']].values.reshape(-1, 1)

polyModel = fitRegression(X3, y3, deg=3)

# %%
#This cell plots the regression for the third movie
plotRegression(X3, y3, polyModel, deg=3)
plt.xlabel('Line Number', fontsize=14)
plt.ylabel('Adjusted Sentiment Value', fontsize=14)
plt.title('Episode VI', loc='center')
plt.show()

# %%
# This cell prints the regression equation for the third movie
print("Predicted Avg Sentiment =", polyModel.intercept_[0], "+",
      polyModel.coef_[0][0], "* (Line #) +", polyModel.coef_[0][1], "* (Line #)^2 +",
      polyModel.coef_[0][2], "* (Line #)^3")

# %%
# This cell calculates the average sentiment for every character w/ more than 20 lines across the three movies
relevantCharacterAverages = findCharacterAverages(full_script, minLines=20)
print(relevantCharacterAverages)

# %%
# This cell plots a bar graph for the average sentiments of the characters
relevantCharacterAverages.plot(kind='bar')
plt.xlabel('Character', fontsize=12)
plt.ylabel('Avg Expressed Sentiment Value', fontsize=12)
plt.title('Sentiments Expressed')
plt.show()

# %%
# This cell splices the movies into dialogues (back-and-forths of length > 4)
dialogues = spliceDialogues(movies, minLength=4)

# %%
# This cell calculates the average received sentiment for every character
characters = [*set(full_script['character'].values.tolist())]
characterAverageReceived = findAverageReceived(dialogues, characters, minLines=20)
print(characterAverageReceived)

# %%
# This cell plots a bar graph for the average received sentiments of the characters
characterAverageReceived.plot(kind='bar')
plt.xlabel('Character', fontsize=12)
plt.ylabel('Avg Received Sentiment Value', fontsize=12)
plt.title('Sentiments Received in Dialogue')
plt.show()

# %%
sentimentTable = joinSentimentSeries(relevantCharacterAverages, characterAverageReceived)

# %%
# This cell graphs the expressed vs. received sentiments
plt.scatter(sentimentTable['expressed'], sentimentTable['received'])
plt.title('Sentiments Expressed vs Sentiments Received')
plt.xlabel('Avg Expressed Sentiment Value')
plt.ylabel('Avg Received Sentiment Value')

for character in sentimentTable['character'].values:
	if character == 'LUKE':
		plt.text(sentimentTable.expressed[sentimentTable.character==character]-0.025,
	  		 	 sentimentTable.received[sentimentTable.character==character]-0.005, 'LUKE')
	elif character == 'JABBA':
		plt.text(sentimentTable.expressed[sentimentTable.character==character]-0.01,
	  		 	 sentimentTable.received[sentimentTable.character==character]+0.01, 'JABBA')
	elif character == 'VADER':
		plt.text(sentimentTable.expressed[sentimentTable.character==character]-0.013,
	  		 	 sentimentTable.received[sentimentTable.character==character]-0.016, 'VADER')
	elif character == 'OWEN':
		plt.text(sentimentTable.expressed[sentimentTable.character==character]-0.012,
	  		 	 sentimentTable.received[sentimentTable.character==character]-0.016, 'OWEN')
	elif character == 'WEDGE':
		plt.text(sentimentTable.expressed[sentimentTable.character==character]-0.033,
	  		 	 sentimentTable.received[sentimentTable.character==character], 'WEDGE')
	else:
		plt.text(sentimentTable.expressed[sentimentTable.character==character]+0.003,
	  		 	 sentimentTable.received[sentimentTable.character==character]+0.003, character)

plt.show()
