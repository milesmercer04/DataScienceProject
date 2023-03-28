// *********************************************************************************************
// Name: Miles Mercer
// Class: CPSC 222, Spring 2023
// Description: This program takes user input in the terminal for high sentiment phrases and the
//              sentiments they express and formats the input into a json file to be read by
//              rasa_nlu in the Python script
// Date Last Modified: 3/2/23
// *********************************************************************************************

#include <iostream>
#include <fstream>
#include <cctype>
#include <algorithm>
#include <vector>
using namespace std;

void generateNewSentimentFile(fstream &, string);
void modifyExistingSentimentFile(fstream &, string);
void makeDataEntry(fstream &, string, bool, bool &);

int main() {
    int modeSelection = 0;
    string userInput, filePath;
    fstream sentimentFile;
    
    cout << "Would you like to make a new file or add on to an existing one?" << endl;
    cout << "1. New File" << endl;
    cout << "2. Existing File" << endl;

    getline(cin, userInput);
    modeSelection = stoi(userInput);
    
    cout << endl << "Enter relative file path" << endl;
    getline(cin, filePath);

    if (modeSelection == 1) {
        generateNewSentimentFile(sentimentFile, filePath);
    } else if (modeSelection == 2) {
        modifyExistingSentimentFile(sentimentFile, filePath);
    } else {
        cout << "INVALID MODE SELECTION" << endl;
        exit(-1);
    }

    return 0;
}

// ************************************************************************************************
// Function: generateNewSentimentFile()
// Description: This function opens up a new json file and takes user input for phrases and their
//              sentiments, outputting them to the file in the correct format for training rasa_nlu
// Input Parameters: An fstream object, sentimentFile, passed by reference, and a string for the
//                   relative file path
// Returns: N/A
// Pre: The file intended to be opened by sentimentFile is either empty or can safely be emptied,
//      and filePath contains a valid relative file path
// Post: User inputs have been outputted to the sentiment file in the standard format
// ************************************************************************************************
void generateNewSentimentFile(fstream &sentimentFile, string filePath) {
    string phrase, userInput;
    bool stillEnteringData = true, isPositiveSentiment, isFirstIteration = true;

    // Open the file specified by filePath
    sentimentFile.open(filePath, ios::out);
    if (!sentimentFile.is_open() || sentimentFile.bad()) {
        cout << "ERROR OPENING FILE" << endl;
        exit(-1);
    } else {
        cout << "FILE OPENED SUCCESSFULLY" << endl << endl;
    }

    // Create "header" portion of json file
    sentimentFile << "{" << endl;
    sentimentFile << "    \"rasa_nlu_data\": {" << endl;
    sentimentFile << "        \"entity_synonyms\": [" << endl;
    sentimentFile << endl;
    sentimentFile << "        ]," << endl;
    sentimentFile << "        \"common_examples\": [" << endl;
    
    // Prompt user for phrases and sentiments, adding these to the list
    while (stillEnteringData) {
        cout << "Enter a high sentiment phrase or \"quit\"" << endl;
        getline(cin, phrase);
        
        // Taken from Stack Overflow, may produce unexpected results
        transform(phrase.begin(), phrase.end(), phrase.begin(),
            [](unsigned char c){ return tolower(c); });

        if (phrase == "quit") {
            stillEnteringData = false;
        } else {
            cout << "Enter 1 for positive sentiment or 0 for negative" << endl;
            getline(cin, userInput);
            isPositiveSentiment = stoi(userInput);
            cout << endl;

            // Format and print the collected data to the JSON file
            makeDataEntry(sentimentFile, phrase, isPositiveSentiment, isFirstIteration);
        }
    }
    
    // Finish off json file
    sentimentFile << endl;
    sentimentFile << "        ]" << endl;
    sentimentFile << "    }" << endl;
    sentimentFile << "}";
}

// ************************************************************************************************
// Function: modifyExistingSentimentFile()
// Description: This function opens up an existing json file and takes user input for phrases and
//              their sentiments, outputting them to the file in the correct format for training
//              rasa_nlu
// Input Parameters: 
void modifyExistingSentimentFile(fstream &sentimentFile, string filePath) {
    vector<string> fileLines;
    string phrase, userInput, currLine;
    bool stillEnteringData = true, isPositiveSentiment, isFirstIteration = false;

    // Open the file specified by filePath in input mode
    sentimentFile.open(filePath, ios::in);
    
    if (!sentimentFile.is_open() || sentimentFile.bad()) {
        cout << "ERROR OPENING FILE" << endl;
        exit(-1);
    } else {
        cout << "FILE OPENED SUCCESSFULLY" << endl << endl;
    }

    // Read all lines of file into string vector
    while (!sentimentFile.eof()) {
        getline(sentimentFile, currLine);
        fileLines.push_back(currLine);
    }

    // Reopen file in output mode and print all lines from vector up until last four lines
    sentimentFile.close();
    sentimentFile.open(filePath, ios::out);
    for (int i = 0; i < fileLines.size() - 4; i++) {
        sentimentFile << fileLines.at(i) << endl;
    }
    fileLines.clear();
    sentimentFile << "            }";
    
    // Prompt user for phrases and sentiments, adding these to the list
    while (stillEnteringData) {
        cout << "Enter a high sentiment phrase or \"quit\"" << endl;
        getline(cin, phrase);
        
        // Taken from Stack Overflow, may produce unexpected results
        transform(phrase.begin(), phrase.end(), phrase.begin(),
            [](unsigned char c){ return tolower(c); });

        if (phrase == "quit") {
            stillEnteringData = false;
        } else {
            cout << "Enter 1 for positive sentiment or 0 for negative" << endl;
            getline(cin, userInput);
            isPositiveSentiment = stoi(userInput);
            cout << endl;

            // Format and print the collected data to the JSON file
            makeDataEntry(sentimentFile, phrase, isPositiveSentiment, isFirstIteration);
        }
    }
    
    // Finish off json file
    sentimentFile << endl;
    sentimentFile << "        ]" << endl;
    sentimentFile << "    }" << endl;
    sentimentFile << "}";
}

void makeDataEntry(fstream &sentimentFile, string phrase, bool isPositiveSentiment,
                   bool &isFirstIteration) {
    if (isFirstIteration) {
        sentimentFile << "            {" << endl;
        sentimentFile << "                \"text\": \"" << phrase << "\"," << endl;
        sentimentFile << "                \"intent\": \"";

        // Inside the quotes for intent, write either positive or negative based on sentiment
        if (isPositiveSentiment) {
            sentimentFile << "positive\"," << endl;
        } else {
            sentimentFile << "negative\"," << endl;
        }

        sentimentFile << "                \"entities\": []" << endl;
        sentimentFile << "            }";
        isFirstIteration = false;
    } else {
        sentimentFile << "," << endl;
        sentimentFile << "            {" << endl;
        sentimentFile << "                \"text\": \"" << phrase << "\"," << endl;
        sentimentFile << "                \"intent\": \"";

        // Inside the quotes for intent, write either positive or negative based on sentiment
        if (isPositiveSentiment) {
            sentimentFile << "positive\"," << endl;
        } else {
            sentimentFile << "negative\"," << endl;
        }

        sentimentFile << "                \"entities\": []" << endl;
        sentimentFile << "            }";
    }
}