"""Generate a txt file using a csv file."""

import pandas as pd
from typing import List

# from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string


def transform(csv_file: str, txt_file: str, cols: List[int]) -> None:
    """Extract the free responses in <csv_file> and store them in <txt_file>.
    """
    porter = PorterStemmer()
    data = pd.read_csv(csv_file, usecols=cols)  # read csv file in the folder
    with open(txt_file, 'a+', encoding='utf-8') as file:
        for col_name in data:  # for the int representing columns in list cols
            for words in data[col_name]:  # for words in each column
                if not pd.isnull(words):  # filter out the null value

                    # preprocessing
                    # lower case
                    words = words.lower()
                    # remove punctuation
                    words = "".join([char for char in words if char not in string.punctuation])
                    # tokenization
                    words = words.split()
                    # remove stop words
                    # stop_words = stopwords.words('english')
                    # filtered_words = [word for word in words if word not in stop_words]
                    # stemming
                    words = [porter.stem(word) for word in words]
                    # rejoin processed words into a str
                    words = " ".join(words)

                    # one row for each response
                    file.write(str(words) + '\n')


if __name__ == '__main__':
    transform('survey.csv', 'survey.txt', [8])
