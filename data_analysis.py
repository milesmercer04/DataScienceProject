# %%
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

# %%
# Description: This cell reads the CSV's for the three movies into separate dataframes and makes an
# 			   additional dataframe for all three movies combined
# Input: The CSV files containing all of the lines in the movies as well as their adjusted sentiment
# 		  values
# Output: Three dataframes (ep1, ep2, & ep3) are created for the individual movie scripts. A list is
# 		  also created containing all three dataframes as well as a single dataframe with all three
# 		  movies concatenated together
# Notes: - This declaration has the disadvantage of using more memory than it needs to. I could have
# 		   restricted myself to only using the list and the concatenated script, but using separate
# 	       dataframes here made the subsequent analysis easier to track
# 		 - You may notice that the use of ep1, ep2, and ep3 doesn't match the names of the CSV files.
# 		   This may be confusing for those unfamiliar with the film franchise, but this naming was in
# 		   keeping with the ordering prior to the creation of the prequels

ep1 = pd.read_csv('scripts/EpisodeIV_Sentiments.csv')
ep2 = pd.read_csv('scripts/EpisodeV_Sentiments.csv')
ep3 = pd.read_csv('scripts/EpisodeVI_Sentiments.csv')

movies = [ep1, ep2, ep3]

full_script = pd.concat(movies)

# %%
# Description: This cell separates the X and Y variables for regression and fits a regression model to the
# 			   first movie
# Input: The inputs for this cell are the 'line' and 'sentiment' columns from the ep1 dataframe
# Output: A cubic (degree 3) polynomial regression is made and fit to the data
# Notes: - In this case, a degree three polynomial regression was made for the data, but this changes
# 		   for subsequent movies. The parameter deg in the fitRegression() function allows me to set the
# 		   degree of the polynomial regression. If I choose to change this in the future, however, I
# 		   should be careful to change it in the call to the plotRegression() function as well.
# 		   Otherwise, the plot will not accurately reflect the model

X1 = ep1[['line']].values.reshape(-1, 1)
y1 = ep1[['sentiment']].values.reshape(-1, 1)

polyModel = fitRegression(X1, y1, deg=3)

# %%
# Description: This cell plots the regression for the first movie
# Input: This cell needs the X and y variables used in the regression as numpy arrays as well as the
# 		 fitted regression model
# Output: The regression is plotted and axis labels and a title are added
# Notes: - I refrained from hard coding the axis titles and main title into the plotRegression() function
# 		   because I wanted to be able to generalize the function to different data, which may need
# 		   different titles

plotRegression(X1, y1, polyModel, deg=3)
plt.xlabel('Line Number', fontsize=14)
plt.ylabel('Adjusted Sentiment Value', fontsize=14)
plt.title('Episode IV', loc='center')
plt.show()

# %%
# Description: This cell prints the first movie's regression equation
# Input: This cell only needs the fitted polynomial regression model
# Output: The coef_ and intercept_ methods are used to find the regression equation, which is then
# 		  printed to the terminal
# Notes: - This regression line doesn't serve much practical purpose, other than outlining a literal
# 		   "plot curve" on the sentiment graph. As such, this equation is not of much use. I simply
# 		   calculated it because I thought the equation might be of interest

print("Predicted Avg Sentiment =", polyModel.intercept_[0], "+",
      polyModel.coef_[0][0], "* (Line #) +", polyModel.coef_[0][1], "* (Line #)^2 +",
      polyModel.coef_[0][2], "* (Line #)^3")

# %%
# Description: This cell fits a regression model to the data from the second movie
# Input: The inputs for this cell are the 'line' and 'sentiment' columns from the ep2 dataframe
# Output: A quadratic (degree 2) polynomial regression is made and fit to the data
# Notes: - As explained earlier, the degree of the polynomial regression depends on the movie. We see
# 		   here that the degree of the ep2 regression is 2, whereas ep1 used a degree 3 regression
# 		 - Because the purpose of these regressions was purely visual, I didn't use any numerical
# 		   methods to verify my polynomial regression choices. If these regressions were to be used
# 		   in more robust models, it would be important to numerically verify that the correct regressions
# 		   were used

X2 = ep2[['line']].values.reshape(-1, 1)
y2 = ep2[['sentiment']].values.reshape(-1, 1)

polyModel = fitRegression(X2, y2, deg=2)

# %%
# Description: This cell plots the regression for the second movie
# Input: This cell needs the X and y variables used in the regression as numpy arrays as well as the
# 		 fitted regression model
# Output: The regression is plotted and axis labels and a title are added
# Notes: - Here we see why th title and axis labels weren't hard coded in the plotRegression() function.
# 		   The previous plot had the title "Episode IV", but this one is titled "Episode V"

plotRegression(X2, y2, polyModel, deg=2)
plt.xlabel('Line Number', fontsize=14)
plt.ylabel('Adjusted Sentiment Value', fontsize=14)
plt.title('Episode V', loc='center')
plt.show()

# %%
# Description: This cell prints the second movie's regression equation
# Input: This cell only needs the fitted polynomial regression model
# Output: The coef_ and intercept_ methods are used to find the regression equation, which is then
# 		  printed to the terminal

print("Predicted Avg Sentiment =", polyModel.intercept_[0], "+",
      polyModel.coef_[0][0], "* (Line #) +", polyModel.coef_[0][1], "* (Line #)^2")

# %%
# Description: This cell fits a regression model to the third movie
# Input: The inputs for this cell are the 'line' and 'sentiment' columns from the ep3 dataframe
# Output: A cubic (degree 3) polynomial regression is made and fit to the data

X3 = ep3[['line']].values.reshape(-1, 1)
y3 = ep3[['sentiment']].values.reshape(-1, 1)

polyModel = fitRegression(X3, y3, deg=3)

# %%
# Description: This cell plots the regression for the third movie
# Input: This cell needs the X and y variables used in the regression as numpy arrays as well as the
# 		 fitted regression model
# Output: The regression is plotted and axis labels and a title are added

plotRegression(X3, y3, polyModel, deg=3)
plt.xlabel('Line Number', fontsize=14)
plt.ylabel('Adjusted Sentiment Value', fontsize=14)
plt.title('Episode VI', loc='center')
plt.show()

# %%
# Description: This cell prints the regression equation for the third movie
# Input: This cell only needs the fitted polynomial regression model
# Output: The coef_ and intercept_ methods are used to find the regression equation, which is then
# 		  printed to the terminal

print("Predicted Avg Sentiment =", polyModel.intercept_[0], "+",
      polyModel.coef_[0][0], "* (Line #) +", polyModel.coef_[0][1], "* (Line #)^2 +",
      polyModel.coef_[0][2], "* (Line #)^3")

# %%
# Description: This cell calculates the average sentiment for every character w/ more than 20 lines across
# 			   the three movies
# Input: This cell needs the full script dataframe (all three movies combined end-to-end)
# Output: The cell calculates the average expressed sentiment for each character with over 20 lines and
# 		  stores them in a pandas Series called relevantCharacterAverages
# Notes: - This Series is easily confused with characterAverageReceived, the average sentiment RECEIVED
# 		   by each character in dialogue. This Series stores the average sentiment EXPRESSED by each
# 		   character

relevantCharacterAverages = findCharacterAverages(full_script, minLines=20)

# %%
# Description: This cell plots a bar graph for the average sentiments of the characters
# Input: Each major charactor's average expressed sentiment has been calculated and stored in
# 		 relevantCharacterAverages, which this cell accesses
# Output: The average expressed sentiments are plotted and the plot is titled and shown
# Notes: - Like the sentiment regressions for each movie, I didn't want to hard code the axis titles and
# 		   main title. In fact, since pandas Series already have a member function to generate a bar plot,
# 		   I didn't need to write my own function for this task at all

relevantCharacterAverages.plot(kind='bar')
plt.xlabel('Character', fontsize=12)
plt.ylabel('Avg Expressed Sentiment Value', fontsize=12)
plt.title('Sentiments Expressed')
plt.show()

# %%
# Description: This cell splices the movies into dialogues (back-and-forths of length >= 4)
# Input: The list of individual movie dataframes called movies
# Output: The movies are spliced into different dialogues, which are stored as individual dataframes in
# 		  the dialogues list
# Notes: - In testing this function, I noticed a strange flaw where sometimes a third character is
# 		   involved in dialogue yet doesn't sleep. One example is when Luke gave Threepio an instruction
# 		   and left the room. Threepio then proceeds to berate Artoo, but Artoo responds only in beeps,
# 		   which aren't included in the script. As a result, my algorithm classifies the interaction as a
# 		   dialogue between Luke and Threepio, despite Luke not being in the room for the majority of it.
# 		   Because I don't have the data to teach my computer to fix this issue, I have to write it off as
# 		   an inherent flaw in the analysis I'm conducting

dialogues = spliceDialogues(movies, minLength=4)

# %%
# Description: This cell calculates the average received sentiment for every character
# Input: This cell needs the list of dialogues generated in the previous cell
# Output: The list of dialogues is used to calculate the average sentiment received by each major character
# 		  in dialogue. These received sentiments are stored in the pandas Series characterAverageReceived.
# 		  Any character who has less than 20 lines directed at them in dialogue is removed from this Series
# Notes: - I was conflicted in coding this process between restricting the Series to characters who have
# 		   more than 20 lines directed at them and just using the list of characters in
# 		   relevantCharacterAverages. Ultimately, I decided this would be the better option, because when
# 		   the two series are combined into a single dataframe, this ensures that all characters have a
# 		   significant number of sentiments expressed and received

characters = [*set(full_script['character'].values.tolist())]
characterAverageReceived = findAverageReceived(dialogues, characters, minLines=20)

# %%
# Description: This cell plots a bar graph for the average received sentiments of the characters
# Input: This cell uses the characterAverageReceived Series generated in the previous cell
# Output: A bar plot is generated for the characters' average received sentiments and displayed
# Notes: - Like the bar plot generated for average expressed sentiments, I didn't create a function to
# 		   generate this graph. This is because pandas Series already have a plot member function

characterAverageReceived.plot(kind='bar')
plt.xlabel('Character', fontsize=12)
plt.ylabel('Avg Received Sentiment Value', fontsize=12)
plt.title('Sentiments Received in Dialogue')
plt.show()

# %%
# Description: This cell combines the Series for characters' average expressed and received sentiments
# 			   into a single dataframe
# Input: This cell uses the relevantCharacterAverages and characterAverageReceived pandas Series generated
# 		 in past cells
# Output: The two Series are turned to dataframes and combined with an inner join. The resulting dataframe
# 		  is called sentimentTable
# Notes: - The use of an inner join results in the loss of a number of characters who appear in one Series
# 		   but not both. This limit in characters, however, turns out to be advantageous in making a graph
# 		   that can be understood visually

sentimentTable = joinSentimentSeries(relevantCharacterAverages, characterAverageReceived)

# %%
# Description: This cell graphs the expressed vs. received sentiments
# Input: This cell uses the sentimentTable dataframe created in the previous cell
# Output: The expressed and received sentiments of major characters are plotted in a scatter plot. The
# 		  points are all labeled with character names
# Notes: - Becuase the labels overlapped when they were algorithmically placed, I moved some of the labels
# 		   around manually to make the graph more readible. This works well for my specific analysis, but
# 		   if the data were to change, these translation would no longer be appropriate. If one wanted to
# 		   reverse the label movements, they would need to delete the conditional in the for loop and
# 		   replace it with only the line in the else clause

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
