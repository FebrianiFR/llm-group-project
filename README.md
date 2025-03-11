# llm-group-project

This repository as the submission for Group Project in Large Language Model Class - MSc Business Analytics - The University of Ediburgh under the guidance of Dr. Tong Wang

## Topic: LLM-Based Fake News Detection Project


## Contents 
- config.json: placeholder API key, to be changed with you open ai api secret
- prompts.json: prompts used in api call
- main.py: main script to run the fakenews detection
- fake_news_classifier.py: class to classify fake news
- hybrid_news_classifier.py: special class to combine different method of fake news detection
- input_data: data used
- output_data: final output of the fake news detection performance across different prompt types and methods


## Source of Data:
### 1. Liar Dataset (https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip)
- This is a tab separated value dataset that contains test, train, and validation split
- The data that will be used in this repo is only the test part
   
### 2. FakeNews Dataset ( https://github.com/KaiDMML/FakeNewsNet)
- Comma separated value, with the file already preprocessed from the original source
- Preprocessing include scrapping article text using the url provided

## How to run the code

### 1. Installing the Libraries
```python
pip install -r requirements.txt
```

### 2. Modify the config.json file to put your OpenAI API Key
"OPENAI_API_KEY": ""

### 3. Run the code
```python
main.py
```

## Key Terms and Definitions
| Terms          | Definition                   |
|----------------|------------------------------|
| Prompt Type  | different prompt type of long and short   |
| Method  | different methods of fake news detections that consist of title, url, and article text <br> special method of hybrid for using title, url, and text in different class  |

## Special notes
- as the article text in the fakenews dataset is scrapped by the authors, some of the sample will not have article text <br> article texts that are missing will fallback to title in the method of article text.
- Requirements for using your own data: the csv must have the column label of real/fake and one of title,news_url,article_text
