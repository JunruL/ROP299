"""Generate a txt file using a csv file."""

import pandas as pd
from typing import List

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string


def transform(csv_file: str, txt_file: str, cols: List[int]) -> None:
    """Extract the free responses in <csv_file> and store them in <txt_file>.
    """
    data = pd.read_csv(csv_file, usecols=cols)  # read csv file in the folder
    with open(txt_file, 'a+', encoding='utf-8') as file:
        for col_name in data:  # for the int representing columns in list cols
            for words in data[col_name]:  # for words in each column
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
    transform('survey.csv', 'GloVe/result1/survey.txt', [8])
