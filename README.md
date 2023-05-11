# DataScienceProject
Semester-long project for Intro to Data Science (CPSC 222)

Contents: 
- All original scripts as well as the scripts with added sentiment and entity columns can be found in the scripts folder
- The CSV file for protagonist and antagonist classification is in the scripts folder
- All graphs from data analysis are in the graphs folder
- The equations from the polynomial regressions are in the equations.txt file

- The technical report is in the Technical Report.docx file
- Sentiment analysis was performed in sentiment_analysis.py
- Data analysis was performed in data_analysis.py
- Classification was performed in classification.py
- Function definitions are in utils.py
- The training data for Rasa NLU is sentiments.json
- The C++ program to make that file is NlpDatasetMaker.cpp
- The executable binary to make sentiment JSON files is a.out

Notes:
- The bulk of the code in sentiment_analysis.py is not mine. My contributions are best found in NlpDatasetMaker.cpp,
  data_analysis.py, classification.py, and utils.py
- If you wish to run sentiment_analysis.py, it needs to be run in Python 3.7. I used a Conda environment

Sources:
- https://www.kaggle.com/datasets/xvivancos/star-wars-movie-scripts
- https://github.com/Macbee280/CrimsonCode2023
