"""Generate a txt file using a csv file."""

import pandas as pd
from typing import List

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

QUESTIONS = {7: '2. What is your preferred mode of teaching for online courses.',
             8: 'Why is this your preferred mode? ',
             26: '10. Do you prefer online or in person courses? ',
             27: '11. Why do you prefer online or in-person courses? '}


def transform(csv_file: str, txt_file: str, cols: List[int]) -> None:
    """Extract the free responses in <csv_file> and store them in <txt_file>.
    """
    data = pd.read_csv(csv_file, usecols=cols)  # read csv file in the folder
    with open(txt_file, 'a+', encoding='utf-8') as file:
        for col_name in data:  # for the int representing columns in list cols
            for words in data[col_name]:  # for words in each column
                if not pd.isnull(words):  # filter out the null value
                    file.write(preprocess(words) + '\n')  # one row for each response


def filter_and_transform(csv_file: str, txt_file: str, cols: List[int], filter: str) -> None:
    """filter the text with cols[0] being <filter>."""
    data = pd.read_csv(csv_file, usecols=cols)  # read csv file in the folder
    filter_col = QUESTIONS[cols[0]]
    data = data.loc[data[filter_col] == filter]
    with open(txt_file, 'a+', encoding='utf-8') as file:
        question_col = QUESTIONS[cols[1]]
        for words in data[question_col]:  # for words in each column
            if not pd.isnull(words):  # filter out the null value
                file.write(preprocess(words) + '\n')  # one row for each response


def preprocess(text: str) -> str:
    """Preprocess <text>."""
    # lower case
    text = text.lower()
    # remove punctuation
    text = "".join([word for word in text if word not in string.punctuation])
    # tokenization
    text = text.split()
    # remove stop words
    stop_words = stopwords.words('english')
    text = [word for word in text if word not in stop_words]
    # stemming
    porter = PorterStemmer()
    text = [porter.stem(word) for word in text]
    # rejoin processed words into a str
    text = " ".join(text)
    return text


if __name__ == '__main__':
    filter_and_transform('coded_data.csv', 'GloVe/col_8_materials/survey_data.txt',
                         [7, 8], 'Uploaded or emailed Materials')
    filter_and_transform('coded_data.csv', 'GloVe/col_8_discussion/survey_data.txt',
                         [7, 8], 'Discussion forums/chats')
