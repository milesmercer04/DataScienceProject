// **********************************************************************************************
// Name: Miles Mercer
// Class: CPSC 222, Spring 2023
// Description: This program takes user input in the terminal for high sentiment phrases, the
// sentiments they express, and the entities they indicate and formats the input into a json file
// to be read by rasa_nlu in the Python script
// Date Last Modified: 3/29/23
// **********************************************************************************************

#include <iostream>
#include <fstream>
#include <cctype>
#include <vector>
using namespace std;

void generateNewSentimentFile(fstream &, string);
void modifyExistingSentimentFile(fstream &, string);
void makeDataEntry(fstream &, string, bool, vector<string>, vector<string>, bool &);
void writeEntitySynonyms(fstream &, ifstream &, string, string);

int findTriggerInPhrase(string, string);

string convertToLower(string);
string convertToUpper(string);

int main() {
    int modeSelection = 0;
    string userInput, filePath, characterListPath;
    fstream sentimentFile;
    ifstream characterList;
    
    cout << "Would you like to make a new file, add on to an existing one, or write entity synonyms?" << endl;
    cout << "1. New File" << endl;
    cout << "2. Existing File" << endl;
    cout << "3. Write Entity Synonyms" << endl;

    getline(cin, userInput);
    modeSelection = stoi(userInput);
    
    cout << endl << "Enter relative file path" << endl;
    getline(cin, filePath);

    if (modeSelection == 1) {
        generateNewSentimentFile(sentimentFile, filePath);
    } else if (modeSelection == 2) {
        modifyExistingSentimentFile(sentimentFile, filePath);
    } else if (modeSelection == 3) {
        cout << "Enter character list path" << endl;
        getline(cin, characterListPath);
        writeEntitySynonyms(sentimentFile, characterList, filePath, characterListPath);
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
    vector<string> entities, entityTriggers;
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
        
        // Convert phrase to lowercase
        phrase = convertToLower(phrase);

        if (phrase == "quit") {
            stillEnteringData = false;
        } else {
            // Get the sentiment of the phrase
            cout << "Enter 1 for positive sentiment or 0 for negative" << endl;
            getline(cin, userInput);
            isPositiveSentiment = stoi(userInput);

            // Get the names of any entities in the line
            entities.clear();
            entityTriggers.clear();
            cout << "Enter the names of any entities in the line, or press enter to continue" << endl;
            getline(cin, userInput);
            while (userInput != "") {
                // Convert user input to uppercase
                userInput = convertToUpper(userInput);
                entities.push_back(userInput);

                // Prompt user for name of entity as it appears in the line
                cout << "Enter the name of the entity as it appears in the line" << endl;
                getline(cin, userInput);

                // Convert entity reference to lowercase
                userInput = convertToLower(userInput);
                entityTriggers.push_back(userInput);

                // Prompt user for next entity
                cout << "Enter any additional entities, or press enter to continue" << endl;
                getline(cin, userInput);
            }
            cout << endl;

            // Format and print the collected data to the JSON file
            makeDataEntry(sentimentFile, phrase, isPositiveSentiment, entities, entityTriggers, isFirstIteration);
            isFirstIteration = false;
        }
    }
    
    // Finish off json file
    sentimentFile << endl;
    sentimentFile << "        ]" << endl;
    sentimentFile << "    }" << endl;
    sentimentFile << "}";

    // Close the sentiment file
    sentimentFile.close();
}

// **********************************************************************************************
// Function: modifyExistingSentimentFile()
// Description: This function opens up an existing json file and takes user input for phrases and
//              their sentiments, outputting them to the file in the correct format for training
//              rasa_nlu
// Input Parameters: An fstream object for the json file and a string for its filepath
// Returns: N/A
// Pre: The filePath string is a valid file path to an existing json file created in the format
//      generated by generateNewSentimentFile()
// Post: The sentiment file is opened and new entries are added in the same format
// **********************************************************************************************
void modifyExistingSentimentFile(fstream &sentimentFile, string filePath) {
    vector<string> fileLines, entities, entityTriggers;
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
        
        // Convert phrase to lowercase
        phrase = convertToLower(phrase);

        if (phrase == "quit") {
            stillEnteringData = false;
        } else {
            // Get the sentiment of the phrase
            cout << "Enter 1 for positive sentiment or 0 for negative" << endl;
            getline(cin, userInput);
            isPositiveSentiment = stoi(userInput);

            // Get the names of any entities in the line
            entities.clear();
            entityTriggers.clear();
            cout << "Enter the names of any entities in the line, or press enter to continue" << endl;
            getline(cin, userInput);
            while (userInput != "") {
                // Convert user input to uppercase
                userInput = convertToUpper(userInput);
                entities.push_back(userInput);

                // Prompt user for name of entity as it appears in the line
                cout << "Enter the name of the entity as it appears in the line" << endl;
                getline(cin, userInput);

                // Convert entity reference to lowercase
                userInput = convertToLower(userInput);
                entityTriggers.push_back(userInput);

                // Prompt user for next entity
                cout << "Enter any additional entities, or press enter to continue" << endl;
                getline(cin, userInput);
            }
            cout << endl;

            // Format and print the collected data to the JSON file
            makeDataEntry(sentimentFile, phrase, isPositiveSentiment, entities, entityTriggers, isFirstIteration);
        }
    }
    
    // Finish off json file
    sentimentFile << endl;
    sentimentFile << "        ]" << endl;
    sentimentFile << "    }" << endl;
    sentimentFile << "}";

    // Close the sentiment file
    sentimentFile.close();
}

// ************************************************************************************************
// Function: makeDataEntry()
// Description: This function adds a single entry to a sentiment file
// Input Parameters: The sentimentFile fstream object by reference, a string for the high sentiment
//                   phrase, a bool for the sentiment (true for positive, false for negative), a
//                   vector of strings for the list of entities in a phrase, and a bool by
//                   reference indicating if this is the first entry to the sentiment file
// Returns: N/A
// Pre: The header portion of the sentiment file has already been printed and the cursor is on the
//      appropriate line for the entry. The isFirstIteration variable is true ONLY if this is the
//      first entry to the file (this is the first iteration of the while loop in
//      generateNewSentimentFile())
// Post: An entry is added to the sentiment file with the appropriate information
// ************************************************************************************************
void makeDataEntry(fstream &sentimentFile, string phrase, bool isPositiveSentiment,
                   vector<string> entities, vector<string> entityTriggers, bool &isFirstIteration) {
    int triggerStartIndex;
    
    if (!isFirstIteration) {
        sentimentFile << "," << endl;
    }
    sentimentFile << "            {" << endl;
    sentimentFile << "                \"text\": \"" << phrase << "\"," << endl;
    sentimentFile << "                \"intent\": \"";

    // Inside the quotes for intent, write either positive or negative based on sentiment
    if (isPositiveSentiment) {
        sentimentFile << "positive\"," << endl;
    } else {
        sentimentFile << "negative\"," << endl;
    }

    sentimentFile << "                \"entities\": [";

    // If there are no entities, close brackets and exit function
    if (entities.size() == 0) {
        sentimentFile << "]" << endl;
        sentimentFile << "            }";
        return;
    }

    if (entities.size() != entityTriggers.size()) {
        cout << "UNEQUAL ENTITIES AND ENTITY TRIGGERS" << endl;
        exit(-1);
    }

    // Insie the brackets for entities, print all the entities in the vector
    sentimentFile << endl;
    for (int i = 0; i < entities.size(); i++) {
        if (i > 0) {
            sentimentFile << "," << endl;
        }
        sentimentFile << "                    {" << endl;
        sentimentFile << "                        \"start\": ";

        triggerStartIndex = findTriggerInPhrase(phrase, entityTriggers.at(i));
        if (triggerStartIndex == -1) {
            cout << "ENTITY TRIGGER NOT FOUND" << endl;
            exit(-1);
        }

        sentimentFile << triggerStartIndex << "," << endl;
        sentimentFile << "                        \"end\": "
                      << (triggerStartIndex + (entityTriggers.at(i)).length()) << "," << endl;
        sentimentFile << "                        \"value\": \"" << entities.at(i) << "\"," << endl;
        sentimentFile << "                        \"entity\": \"character\"" << endl;
        sentimentFile << "                    }";
    }

    sentimentFile << endl;
    sentimentFile << "                ]" << endl;
    sentimentFile << "            }";
}

// **********************************************************************************************
// Function: writeEntitySynonyms()
// Description: This function prompts the user for synonyms for all character names in Star Wars,
//              formatting the responses and putting them in the entity synonyms section of the
//              sentiment file
// Input Parameters: The sentiment file, a file with the list of character names, a string
//                   with the filepath for the sentiment file, and a string with the filepath for
//                   the character list
// Returns: N/A
// Pre: The sentiment file does not yet contain any entity synonym entries. The character list is
//      every character in the scripts. Both files' relative paths are accurately reflected by
//      their respective strings
// Post: The entity synonyms section is written to the sentiment file and the old contents are
//       pasted back after the section
// **********************************************************************************************
void writeEntitySynonyms(fstream &sentimentFile, ifstream &characterList, string filePath, string characterListPath) {
    vector<string> fileLines, characters;
    string currLine, synonym;
    bool firstSynonym;
    
    // Open the file specified by filePath in input mode
    sentimentFile.open(filePath, ios::in);
    
    if (!sentimentFile.is_open() || sentimentFile.bad()) {
        cout << "ERROR OPENING SENTIMENT FILE" << endl;
        exit(-1);
    } else {
        cout << "SENTIMENT FILE OPENED SUCCESSFULLY" << endl;
    }

    // Open the character list file specified by characterListPath
    characterList.open(characterListPath);

    if (!characterList.is_open() || characterList.bad()) {
        cout << "ERROR OPENING CHARACTER FILE" << endl;
        exit(-1);
    } else {
        cout << "CHARACTER FILE OPENED SUCCESSFULLY" << endl << endl;
    }

    // Read all lines of file into string vector
    while (!sentimentFile.eof()) {
        getline(sentimentFile, currLine);
        fileLines.push_back(currLine);
    }

    // Read all characters into a string vector
    while (!characterList.eof()) {
        getline(characterList, currLine);
        characters.push_back(currLine);
    }

    // Close the file and reopen it in output mode, printing out the first three lines
    sentimentFile.close();
    sentimentFile.open(filePath, ios::out);
    for (int i = 0; i < 3; i++) {
        sentimentFile << fileLines.at(i) << endl;
    }

    // For each character, ask user for any synonyms or to press enter, then make an entry
    for (int i = 0; i < characters.size(); i++) {
        if (i > 0) {
            sentimentFile << "," << endl;
        }
        sentimentFile << "            {" << endl;
        sentimentFile << "                \"value\": \"" << characters.at(i) << "\"," << endl;
        sentimentFile << "                \"synonyms\": [";

        cout << "Enter any synonyms for " << characters.at(i) << " or press enter" << endl;
        getline(cin, synonym);
        firstSynonym = true;
        while (synonym != "") {
            if (!firstSynonym) {
                sentimentFile << ", ";
            }
            sentimentFile << "\"" << synonym << "\"";
            cout << "Enter any additional synonyms for " << characters.at(i) << " or press enter" << endl;
            getline(cin, synonym);
            firstSynonym = false;
        }
        sentimentFile << "]" << endl;
        sentimentFile << "            }";
    }

    // Print remaining lines from original sentimentFile starting at line 5
    sentimentFile << endl;
    for (int i = 4; i < fileLines.size(); i++) {
        sentimentFile << fileLines.at(i) << endl;
    }

    // Close both files
    sentimentFile.close();
    characterList.close();
}

// ***********************************************************************************************
// Function: findTriggerInPhrase()
// Description: This function takes in a phrase and a trigger phrase, and attempts to find the
//              trigger phrase in the phrase, returning the starting index of the trigger if found
//              or -1 otherwise
// Input Paramters: The larger phrase and the trigger phrase, both as string objects
// Returns: An int for the starting index of the trigger phrase
// Pre: Two phrases are passed into the function
// Post: If the trigger is found, the starting index is returned. Otherwise, -1 is returned
// ***********************************************************************************************
int findTriggerInPhrase(string phrase, string trigger) {
    bool phrasesMatch = false;
    for (int i = 0; i < (phrase.length() - trigger.length() + 1); i++) {
        if (phrase.at(i) == trigger.at(0)) {
            phrasesMatch = true;
            for (int j = 1; j < trigger.length() && phrasesMatch; j++) {
                if (phrase.at(i + j) != trigger.at(j)) {
                    phrasesMatch = false;
                }
            }
            if (phrasesMatch) {
                return i;
            }
        }
    }
    return -1;
}

// ***********************************************************************************************
// Function: convertToLower()
// Description: This function converts a string to lowercase
// Input Parameters: A string to be converted to lowercase
// Returns: The original string in lowercase
// Pre: The string contains some alphabetical characters in uppercase (for the fxn to do anything)
// Post: The original string is converted to lowercase in a new string, which is returned
// ***********************************************************************************************
string convertToLower(string original) {
    string lowerString;
    for (char c : original) {
        lowerString += tolower(c);
    }
    return lowerString;
}

// ***********************************************************************************************
// Function: convertToUpper()
// Description: This function converts a string to uppercase
// Input Parameters: A string to be converted to uppercase
// Returns: The original string in uppercase
// Pre: The string contains some alphabetical characters in lowercase (for the fxn to do anything)
// Post: The original string is converted to uppercase in a new string, which is returned
// ***********************************************************************************************
string convertToUpper(string original) {
    string upperString;
    for (char c : original) {
        upperString += toupper(c);
    }
    return upperString;
}