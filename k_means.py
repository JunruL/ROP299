"""This file is used for k-means clustering."""
import pandas as pd
from sklearn.cluster import KMeans

QUESTIONS = {8: 'Why is this your preferred mode? ',
             18: 'If you could change one thing about the way your online classes are designed, '
                 'what would you change? Why?',
             25: 'If you could change one thing about the way your in-person classes are designed, '
                 'what would you change? Why?',
             27: 'Why do you prefer online or in person courses?'
             }


def key_words_to_vectors(key_words_file: str, vectors_file: str, results_file: str) -> None:
    """Find the vector representations for key words in <key_words_file>, using the
    vectors from <vectors_file>, and store the results in <results_file>.

    Note: This is the initial version with only one key word

    Preconditions:
      - key_words_file is a txt file
      - results_file and vectors_file are csv files
    """
    # TODO: Modify the function so that it can be used for the situation of more than one key word
    word_vectors = pd.read_csv(vectors_file)
    df = pd.DataFrame()
    with open(key_words_file, 'r') as f:
        for line in f.readlines():
            key_word = line[2:-3]  # '['hustl']'[2:-3] returns 'hustl'
            vector = word_vectors.loc[word_vectors['0'] == key_word]
            df = df.append(vector, ignore_index=True)
    f.close()
    df.to_csv(results_file, index=False)


def clustering(original_data: str, key_words_file: str, results_file: str,
               k: int, col: int) -> None:
    """Do clustering using k-means.
    Preconditions:
      - key_words_file and original_data_file are csv files
      - results_file is a txt file
    """
    question = QUESTIONS[col]
    original_df = pd.read_csv(original_data, usecols=[col])
    # filter out null values
    non_empty_data = original_df[original_df[question].notnull()][question].reset_index(drop=True)
    key_words_df = pd.read_csv(key_words_file)
    x = key_words_df.iloc[:, 1:51].values
    k_means = KMeans(n_clusters=k)
    y_k_means = k_means.fit_predict(x)
    result = list(y_k_means)
    key_words = key_words_df['0']
    with open(results_file, 'a+', encoding='utf-8') as file:
        for i in range(len(result)):
            file.write(str(non_empty_data[i]) + ' ' +
                       str(key_words[i]) + ' ' + str(result[i]) + '\n')


def txt_to_csv(txt_file: str, csv_file: str) -> None:
    """Convert a txt file to a csv file.
    Each line in <txt_file> will be a row in <csv_file>, with each column being the values
    separated by ' ' in <txt_file>.
    Preconditions:
      - txt_file is a txt file
      - csv_file is a csv file
    """
    df = pd.read_csv(txt_file, delimiter=' ', header=None)
    df.to_csv(csv_file, index=False)  # Avoid index


if __name__ == '__main__':
    txt_to_csv('GloVe/result1/vectors.txt', 'vectors.csv')
    key_words_to_vectors('key_word.txt', 'vectors.csv', 'key_word_vector.csv')
    clustering('coded_data.csv', 'key_word_vector.csv', 'k_means_results.txt', 5, 8)
