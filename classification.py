# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import metrics
from utils import *

# %%
# Description: This cell reads in the dataframes from the CSV files and cuts them to only the columns we
#              need
# Input: The CSV files for the scripts of the three movies with sentiments as well as the file with
#        classifications of characters as protagonists, antagonists, or neither
# Output: The files are all read into separate dataframes. The episode dataframes have all columns removed
#         except for 'character' and 'sentiment'
# Notes: - The Protagonist_or_Antagonist.csv file is only used in this part of the project. It was not
#          from the original Kaggle dataset. In fact, I made it by hand using an auto-generated list of
#          of all characters in the three movies and Google. Here, it gives us something nice to try to
#          classify with different models

ep1 = pd.read_csv('scripts/EpisodeIV_Sentiments.csv')
ep2 = pd.read_csv('scripts/EpisodeV_Sentiments.csv')
ep3 = pd.read_csv('scripts/EpisodeVI_Sentiments.csv')

ep1 = ep1[['character', 'sentiment']]
ep2 = ep2[['character', 'sentiment']]
ep3 = ep3[['character', 'sentiment']]

classifications = pd.read_csv('scripts/Protagonist_or_Antagonist.csv')

# %%
# Description: This cell merges the episode dataframes with the character classification dataframe and
#              equalizes protagonists and antagonists
# Input: The individul movie dataframes (ep1, ep2, ep3) and the classifications dataframe
# Output: The episode dataframes with an added 'classification' column classifying the character who speaks
#         each line as well as a full_script dataframe with all of these movie dataframes put together
# Notes: - Because the protagonists speak so much more than the antagonists in these movies, balanceScript()
#          samples the protagonist lines randomly so that there is an equal number of protagonist and
#          antagonist lines and none from character classified as "Neither"

ep1 = ep1.merge(classifications, on='character')
ep2 = ep2.merge(classifications, on='character')
ep3 = ep3.merge(classifications, on='character')

full_script = pd.concat([ep1, ep2, ep3])
modified_script = balanceScript(full_script)

# %%
# Description: This cell initializes a KNN classifier and tests it on the data
# Input: The modified_script dataframe created in the previous cell. The 'sentiment' and 'classification'
#        columns are used as the independent and dependent variables for classification
# Output: A KNN Classifier model is initialized and fit to the data from the modified full script. This
#         function is created with training data and tested on separate test data
# Notes: - It will, no doubt, be incredibly obvious when this cell is run that KNN classification is not
#          effective at all. Neither, as it turns out, is a decision tree. I have come to the conclusion
#          that my hypothesis that protagonists would tend toward higher sentiments and antagonists to
#          lower ones was not accurate. I believe that protagonists are written to cover a wider range of
#          sentiments because their greater screen time gives them more opportunity to display a fuller
#          emotional range. Displaying such an emotional range in antagonists may also make them more
#          liked by the audience, which may not be the intention. As a result, it seems that protaginists
#          tended to have more sentiment range, but there were no ranges that were uniquely protagonist or
#          antagonist. Sentiment just doesn't seem to predict protagonist or antagonist classification

knnClassifier = KNeighborsClassifier(n_neighbors=2)
X = modified_script[['sentiment']]
y = modified_script[['classification']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knnClassifier.fit(X_train, np.ravel(y_train))
y_pred = knnClassifier.predict(X_test)
print('KNN Accuracy Score:', metrics.accuracy_score(y_pred, y_test))

# %%
# Description: This cell visualizes the predictions of the KNN classifier
# Input: The fitted KNN Classifier model
# Output: The probabilities of a "Protagonist" classification across the entire possible sentiment range
#         are plotted. The axes are labeled and the graph is titled before being displayed
# Notes: - This graph displays the notes given in the comments for the previous cell. It also gives some
#          visual reason for why the model does such a poor job of predicting protagonist vs. antagonist
#          classification

visualizeKnnClassifier(knnClassifier, -1, 1)
plt.xlabel('Line Sentiment Value')
plt.ylabel('Probability of "Protagonist" Label')
plt.title('KNN Classification Results')
plt.show()

# %%
# Description: This cell initializes a Decision Tree classifier and fits it to the data, testing its accuracy
# Input: The training and testing data generated previously for the KNN classifier
# Output: A decision tree classifier is fitted to the training data and its accuracy is tested on the test
#         data. The tree is then displayed
# Notes: - As with the KNN model, this model does a very poor job of preicting protagonist vs. antagonist
#          classification. There doesn't really seem to be any correlation between expressed sentiment and
#          classification at all.
#        - In the future, it may be interesting to lower the restrictions on calculating average received
#          sentiment so that the data can be used to see if received sentiment is more predictive of
#          protaginist vs. antagonist classification. That may be more helpful

classTreeModel = DecisionTreeClassifier(max_depth=2)
classTreeModel.fit(X_train, np.ravel(y_train))
print(export_text(classTreeModel, feature_names=X_train.columns.to_list()))
print('DT Accuracy Score:', metrics.accuracy_score(y_test, classTreeModel.predict(X_test)))
