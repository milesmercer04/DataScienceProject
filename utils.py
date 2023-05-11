import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statistics import mean

# ************************************************************************************************
# Function: fitRegression()
# Description: This function fits a polynomial regression to an X and y variable and returns the
#              model
# Input Parameters: X, the independent variable stored as a numpy array, y, the dependent variable
#                   stored as a numpy array, and an int for the degree of the desired regression
# Returns: The fitted regression model
# Pre: X and y were both lists but have been reshaped on (-1, 1) before being passed to the
#      function
# Post: A LinearRegression() model is created and fitted to the data based on polynomial features
#       specified by the deg parameter. This model is returned once fitted
# ************************************************************************************************
def fitRegression(X, y, deg):
    polyFeatures = PolynomialFeatures(degree=deg, include_bias=False)
    xPoly = polyFeatures.fit_transform(X)
    polyModel = LinearRegression()
    polyModel.fit(xPoly, y)
    return polyModel

# ***********************************************************************************************
# Function: plotRegression()
# Description: This function plots a polynomial regression using pyplot
# Input Parameters: X and y, the independent and dependent variables passed as numpy arrays, the
#                   LinearRegression() model, and the degree of the polynomial regression
# Returns: N/A
# Pre: X and y are both stored as numpy arrays reshaped on (-1, 1), the model has been fit to the
#      data based on polynomial features specified by the degree, and the degree accurately
#      reflects the degree of the polynomial features used in fitting the model
# Post: The data and regression line are plotted. The plot is not yet shown because it hasn't had
#       its axes labeled or a title added. This is left to the user
# ***********************************************************************************************
def plotRegression(X, y, model, deg):
    polyFeatures = PolynomialFeatures(degree=deg, include_bias=False)
    plt.scatter(X, y, color='black')
    xDelta = np.linspace(X.min(), X.max(), 1000)
    yDelta = model.predict(polyFeatures.fit_transform(xDelta.reshape(-1, 1)))
    plt.plot(xDelta, yDelta, color='blue', linewidth=2)

# *************************************************************************************************
# Function: findCharacterAverages()
# Description: This function calculates the average expressed sentiment for all characters above a
#              set threshold for number of lines
# Input Parameters: The dataframe containing the script to be analyzed and the minimum number of
#                   lines for a characer to be in the list
# Returns: A pandas Series with all characters having more than the specified number of lines
#          sorted by average expressed sentiment in descending order
# Pre: The script dataframe is formatted as the sentiment scripts with sentiments have been in this
#      project have been, or at least has the columns 'character' and 'sentiment'
# Post: The characters' average expressed sentiments are calculated and stored in a descending
#       pandas Series, then all characters with less than the threshold number of lines are removed
#       before the Series is returned
# *************************************************************************************************
def findCharacterAverages(script, minLines=20):
    characterAverages = script.groupby('character')['sentiment'].mean()
    characterAverages.sort_values(ascending=False, inplace=True)

    numLines = {}
    characters = [*set(script['character'].values.tolist())]
    for character in characters:
        numLines[character] = script['character'].value_counts()[character]

    minorCharacters = []
    for character in numLines:
        if (numLines[character] < minLines):
            minorCharacters.append(character)
    
    return characterAverages.drop(labels=minorCharacters)

# ************************************************************************************************
# Function: spliceDialogues()
# Description: This function takes in a list of script dataframes and cuts them up into separate
#              dataframes for each dialogue over a set length
# Input Parameters: The list of scripts and the minimum length of a dialogue to be included in the
#                   list
# Returns: A list of dataframes for each dialogue
# Pre: The movie dataframes have their character column labeled 'character'
# Post: The movie scripts are spliced into dialogues, which are put in a list and the list is
#       returned
# ************************************************************************************************
def spliceDialogues(movies, minLength=4):
    dialogues = []
    for movie in movies:
        numRows = len(movie.index)
        start = 0
        stop = 0

        while stop < numRows - 1:
            characterA = movie.at[start, 'character']
            characterB = movie.at[start + 1, 'character']

            stop = start + 2
            while (stop < numRows and (movie.at[stop, 'character']==characterA or movie.at[stop, 'character']==characterB)):
                stop += 1

            if stop - start >= minLength:
                dialogues.append(movie.iloc[start:stop, :])

            start = stop - 1
    
    return dialogues

# *************************************************************************************************
# Function: findAverageReceived()
# Description: This function calculates the average received sentiment for each character spoken to
#              more than a set number of times in dialogue
# Input Parameters: An array of dataframes containing separate dialogues between characters, a list
#                   of all characters to be analyzed, and the minimum number of lines to be spoken
#                   to each character in dialogue
# Returns: A pandas Series with each major character's average received sentiment
# Pre: The scripts have been spliced into dialogues in the dialogues list and the characters list
#      accurately reflects the names of all the character in the dialogues
# *************************************************************************************************
def findAverageReceived(dialogues, characters, minLines=20):
    characterAverageReceivedDict = {}
    for character in characters:
        receivedSentiments = []
        for dialogue in dialogues:
            if not np.any(dialogue['character'].values == character):
                continue

            for i in dialogue.index:
                if dialogue.at[i, 'character'] != character:
                    receivedSentiments.append(dialogue.at[i, 'sentiment'])

        if len(receivedSentiments) >= minLines:
            characterAverageReceivedDict[character] = mean(receivedSentiments)

    characterAverageReceived = pd.Series(characterAverageReceivedDict, name='sentiment')
    characterAverageReceived.sort_values(ascending=False, inplace=True)
    return characterAverageReceived

# **********************************************************************************************
# Function: joinSentimentSeries()
# Description: This function takes in the pandas Series for the average expressed and received
#              sentiments of major characters and performs an inner join to turn them into one
#              pandas Dataframe, which is returned
# Input Parameters: The pandas Series for the average expressed and received sentiments
# Returns: The dataframe with the combined data
# Pre: Both Series are using the character names as the keys for the sentiments. The character
#      names are consistent between the two Series
# Post: The Series are joined into one Dataframe with column names 'character', 'expressed', and
#       'received'. This Datafram is then returned
# **********************************************************************************************
def joinSentimentSeries(expressed, received):
    expressedDF = expressed.to_frame()
    expressedDF.columns = ['expressed']

    receivedDF = received.to_frame()
    receivedDF.columns = ['received']

    sentimentTable = pd.concat([expressedDF, receivedDF], axis=1, join='inner')
    sentimentTable.reset_index(inplace=True)
    sentimentTable = sentimentTable.rename(columns={'index':'character'})
    return sentimentTable

# *************************************************************************************************
# Function: balanceScript()
# Description: This function takes in a pandas Dataframe containing a script with character
#             classifications and modifies it so that there are no characters classified as
#             'Neither' and an equal number of character classified as 'Proagonist' and
#             'Antagonist'
# Input Parameters: The pandas Dataframe containing the script to be modified
# Returns: A pandas Dataframe with the modified script
# Pre: The script has a column named 'classification' specifying if the character speaking each
#      line is a 'Protagonist', 'Antagonist', or 'Neither'
# Post: The script is modified to have an equal number of protagonist and antagonist lines and none
#       from character classified as neither. This modified script is then returned
# *************************************************************************************************
def balanceScript(script):
    script.drop(script[script['classification'] == 'Neither'].index, inplace=True)
    protagonists = script[script['classification'] == 'Protagonist']
    antagonists = script[script['classification'] == 'Antagonist']

    if len(protagonists) == len(antagonists):
        return script
    elif len(protagonists) > len(antagonists):
        cut_protagonists = protagonists.sample(n=len(antagonists.index))
        return pd.concat([cut_protagonists, antagonists])
    else:
        cut_antagonists = antagonists.sample(n=len(protagonists.index))
        return pd.concat([protagonists, cut_antagonists])

# ************************************************************************************************
# Function: visualizeKnnClassifier()
# Description: This function plots the predictions of a KNN classifier on a range of inputs
# Input Parameters: The KNN classification model and the lower and upper bounds for inputs to be
#                   tested
# Returns: N/A
# Pre: This function assumes that there are only two classifications for the KNN classification
#      model to choose from and that the second is the one to plot
# Post: The likelihood of the KNN Classifier selecting the second classification is calculated for
#       50 x-values in the specified range and plotted in a scatter plot
# ************************************************************************************************
def visualizeKnnClassifier(knnClassifier, minX, maxX):
    xCoors = []
    yCoors = []

    x = minX
    while x <= maxX:
        xCoors.append(x)
        yCoors.append(knnClassifier.predict_proba([[x]])[0, 1])
        x += (maxX - minX) / 50

    plt.scatter(xCoors, yCoors)