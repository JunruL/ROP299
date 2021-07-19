"""Extract key words using TF-IDF.
Tutorial:
https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.YO_hGC-1GMJ
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string


def preprocess(text: str) -> str:
    """Preprocess <text>."""
    if pd.isnull(text):
        return ''
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


def sort_coo(coo_matrix) -> list:
    """..."""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, top_n=1) -> dict:
    """get the feature names and tf-idf score of top n items"""

    # use only top_n items from vector
    sorted_items = sorted_items[:top_n]

    score_values = []
    feature_values = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_values.append(round(score, 3))
        feature_values.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_values)):
        results[feature_values[idx]] = score_values[idx]

    return results


def extract_key_word(csv_file: str, txt_file: str, col: int, max_df: float = 1, top_n=1) -> dict:
    """Create Vocabulary and Word Counts for IDF."""
    cv = CountVectorizer(max_df=max_df)
    # read csv into a dataframe
    df_idf = pd.read_csv(csv_file, usecols=[col])
    col_name = df_idf.columns[0]
    df_idf[col_name] = df_idf[col_name].apply(lambda x: preprocess(x))
    docs = df_idf[col_name].tolist()
    word_count_vector = cv.fit_transform(docs)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    # read test docs into a dataframe and concatenate title and body
    df_test = pd.read_csv(csv_file, usecols=[col])
    col_name = df_test.columns[0]
    df_test[col_name] = df_test[col_name].apply(lambda x: preprocess(x))
    # get test docs into a list
    docs_test = df_test[col_name].tolist()
    # a mapping of index to
    feature_names = cv.get_feature_names()

    with open(txt_file, 'a+', encoding='utf-8') as file:
        # get the document that we want to extract keywords from
        for doc in docs_test:
            # generate tf-idf for the given document
            tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
            # sort the tf-idf vectors by descending order of scores
            sorted_items = sort_coo(tf_idf_vector.tocoo())
            # extract only the top n
            keywords = extract_topn_from_vector(feature_names, sorted_items, top_n)
            if keywords:  # skip the empty one
                file.write(str(list(keywords.keys())) + '\n')  # one row for each response

    return keywords


if __name__ == '__main__':
    extract_key_word('coded_data.csv', 'key_word.txt', 8, 0.85, 1)
