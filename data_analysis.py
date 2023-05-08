# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import r_regression
from sklearn.preprocessing import PolynomialFeatures
from statistics import mean

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

quadraticPolyFeatures = PolynomialFeatures(degree=2, include_bias=False)
cubicPolyFeatures = PolynomialFeatures(degree=3, include_bias=False)
quarticPolyFeatures = PolynomialFeatures(degree=4, include_bias=False)
xPoly1 = cubicPolyFeatures.fit_transform(X1)
polyModel = LinearRegression()
polyModel.fit(xPoly1, y1)

# %%
# This cell plots the regression for the first movie
plt.scatter(X1, y1, color='black')
# plt.plot(X, yPred, color='blue', linewidth=2)
xDelta1 = np.linspace(X1.min(), X1.max(), 1000)
yDelta1 = polyModel.predict(cubicPolyFeatures.fit_transform(xDelta1.reshape(-1, 1)))
plt.plot(xDelta1, yDelta1, color='blue', linewidth=2)
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

xPoly2 = quadraticPolyFeatures.fit_transform(X2)
polyModel.fit(xPoly2, y2)

# %%
# This cell plots the regression for the second movie
plt.scatter(X2, y2, color='black')
xDelta2 = np.linspace(X2.min(), X2.max(), 1000)
yDelta2 = polyModel.predict(quadraticPolyFeatures.fit_transform(xDelta2.reshape(-1, 1)))
plt.plot(xDelta2, yDelta2, color='blue', linewidth=2)
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

xPoly3 = cubicPolyFeatures.fit_transform(X3)
polyModel.fit(xPoly3, y3)

# %%
#This cell plots the regression for the third movie
plt.scatter(X3, y3, color='black')
# plt.plot(X, yPred, color='blue', linewidth=2)
xDelta3 = np.linspace(X3.min(), X3.max(), 1000)
yDelta3 = polyModel.predict(cubicPolyFeatures.fit_transform(xDelta3.reshape(-1, 1)))
plt.plot(xDelta3, yDelta3, color='blue', linewidth=2)
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
# This cell calculates the average sentiment for every character across the three movies
characterAverages = full_script.groupby('character')['sentiment'].mean()
characterAverages.sort_values(ascending=False, inplace=True)
print(characterAverages)

# %%
# This cell removes any characters from the list of average sentiments with less than 20 lines
numLines = {}
characters = [*set(full_script['character'].values.tolist())]
for character in characters:
    numLines[character] = full_script['character'].value_counts()[character]

irrelevantCharacters = []
for character in numLines:
    if (numLines[character] < 20):
        irrelevantCharacters.append(character)

relevantCharacterAverages = characterAverages.drop(labels = irrelevantCharacters)
print(relevantCharacterAverages)

# %%
# This cell plots a bar graph for the average sentiments of the characters
relevantCharacterAverages.plot(kind='bar')
plt.xlabel('Character', fontsize=12)
plt.ylabel('Avg Adjusted Sentiment Value', fontsize=12)
plt.title('Characters w/ 20+ Lines')
plt.show()

# %%
# This cell splices the movies into dialogues (back-and-forths of length > 4)
dialogues = []
for movie in movies:
	numRows = len(movie.index)
	start = 0
	stop = 0

	while (stop < numRows - 1):
		characterA = movie.at[start,'character']
		characterB = movie.at[(start + 1),'character']

		stop = start + 2
		while (stop < numRows and (movie.at[stop,'character'] == characterA or movie.at[stop,'character'] == characterB)):
			stop += 1
		
		if (stop - start >= 4):
			dialogues.append(movie.iloc[start:stop,:])
		
		start = stop - 1 

# %%
# This cell calculates the average received sentiment for every character
characterAverageReceivedDict = {}
for character in characters:
	receivedSentiments = []
	for dialogue in dialogues:
		if not np.any(dialogue['character'].values == character):
			continue

		for i in dialogue.index:
			if dialogue.at[i,'character'] != character:
				receivedSentiments.append(dialogue.at[i,'sentiment'])

	if len(receivedSentiments) > 20:
		characterAverageReceivedDict[character] = mean(receivedSentiments)

characterAverageReceived = pd.Series(characterAverageReceivedDict)
characterAverageReceived.sort_values(ascending=False, inplace=True)
print(characterAverageReceived)

# %%
