# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import metrics, tree

# %%
# This cell reads in the dataframes from the CSV file and cuts them to only the columns we need
ep1 = pd.read_csv('scripts/EpisodeIV_Sentiments.csv')
ep2 = pd.read_csv('scripts/EpisodeV_Sentiments.csv')
ep3 = pd.read_csv('scripts/EpisodeVI_Sentiments.csv')

ep1 = ep1[['character', 'sentiment']]
ep2 = ep2[['character', 'sentiment']]
ep3 = ep3[['character', 'sentiment']]

classifications = pd.read_csv('scripts/Protagonist_or_Antagonist.csv')

# %%
# This cell merges the episode dataframes with the character classification dataframe and equalizes protagonists and antagonists
ep1 = ep1.merge(classifications, on='character')
ep2 = ep2.merge(classifications, on='character')
ep3 = ep3.merge(classifications, on='character')

full_script = pd.concat([ep1, ep2, ep3])
full_script.drop(full_script[full_script['classification'] == 'Neither'].index, inplace=True)

protagonists = full_script[full_script['classification'] == 'Protagonist']
antagonists = full_script[full_script['classification'] == 'Antagonist']

cut_protagonists = protagonists.sample(n=len(antagonists.index))
modified_script = pd.concat([cut_protagonists, antagonists])

# %%
# This cell initializes a KNN classifier and tests it on the data
knnClassifier = KNeighborsClassifier(n_neighbors=2)
X = modified_script[['sentiment']]
y = modified_script[['classification']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knnClassifier.fit(X_train, np.ravel(y_train))
y_pred = knnClassifier.predict(X_test)
print(metrics.accuracy_score(y_pred, y_test))

# %%
# This cell visualizes the predictions of the KNN classifier
xCoors = []
yCoors = []

x = -1
while x <= 1:
    xCoors.append(x)
    yCoors.append(knnClassifier.predict_proba([[x]])[0, 1])
    x += 0.05

plt.scatter(xCoors, yCoors)
plt.xlabel('Line Sentiment Value')
plt.ylabel('Probability of "Protagonist" Label')
plt.title('KNN Classification Results')
plt.show()

# %%
# This cell initializes a Decision Tree classifier and fits it to the data, testing its accuracy
classTreeModel = DecisionTreeClassifier(max_depth=2)
classTreeModel.fit(X_train, np.ravel(y_train))
print(export_text(classTreeModel, feature_names=X_train.columns.to_list()))
print("Accuracy Score:", metrics.accuracy_score(y_test, classTreeModel.predict(X_test)))
