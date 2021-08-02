""""tf-idf and k-means."""

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from typing import Tuple
import string
import os
import random


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


def csv_to_txt(csv_file: str, txt_file: str, col: int) -> None:
    """Generate a text file to store all the responses of in a certain column of csv_file."""
    data = pd.read_csv(csv_file, usecols=[col])
    with open(txt_file, 'a+', encoding='utf-8') as file:
        for doc in data.values:
            if not pd.isnull(doc):
                doc = str(doc)
                doc = doc.replace("['", "")
                doc = doc.replace("']", "")
                doc = doc.replace('["', '')
                doc = doc.replace('"]', '')
                file.write(doc + '\n')
    file.close()


def generate_corpus(txt_file: str) -> Tuple[list, list]:
    """Return a tuple of lists with the first list being the original corpus of
    <txt_file> and the second being the preprocessed corpus.
    """
    corpus = []
    preprocessed_corpus = []
    with open(txt_file, encoding="utf-8") as file:
        for lin in file:
            # remove "\n"
            lin = lin.replace("\n", "")

            corpus.append(lin)
            lin = preprocess(lin)
            preprocessed_corpus.append(lin)
    file.close()
    return (corpus, preprocessed_corpus)


def tf_idf(preprocessed_corpus: list) -> Tuple[list, list]:
    """Return a weight matrix (list of list) and a key word list using tf-idf algorithm."""
    # convert the words into a word frequency matrix
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(preprocessed_corpus)
    feature_names = cv.get_feature_names()

    transformer = TfidfTransformer()
    # convert the word frequency matrix into a TF-IDF matrix
    tfidf = transformer.fit_transform(word_count_vector)
    weight = tfidf.toarray()

    # extracting the key word of each response, which is the word with thne largest weight
    key_words = []
    for doc in preprocessed_corpus:
        # generate tf-idf for the given document
        tf_idf_vector = transformer.transform(cv.transform([doc]))
        # sort the tf-idf vectors by descending order of scores
        coo_matrix = tf_idf_vector.tocoo()
        tuples = zip(coo_matrix.col, coo_matrix.data)
        sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
        if sorted_items == []:
            # There seems to be some bugs, since sorted_items could be empty
            key_words.append('NA')
        else:
            idx = sorted_items[0][0]
            key_words.append(feature_names[idx])

    return (weight, key_words)


def k_means(n: int, weight: list) -> tuple[KMeans, list]:
    """Do clustering using the k-means algorithm.
    n: number of clusters
    """
    k_means = KMeans(n_clusters=n, random_state=0).fit(weight)
    centroid_list = k_means.cluster_centers_  # centers of clustering
    labels = k_means.labels_  # labels of clustering
    n_clusters_ = len(centroid_list)

    cluster_members_list = []
    for i in range(0, n_clusters_):
        members_list = []
        for j in range(0, len(labels)):
            if labels[j] == i:
                members_list.append(j)
        cluster_members_list.append(members_list)

    return (k_means, cluster_members_list)


def generate_graph(weight: list, k_means: KMeans, labels: list, num: int) -> None:
    """Generate a graph based on the k-means clustering result.
    Use TSNE algorithm to reduce the dimensionality.
    """
    tsne = TSNE(n_components=2)
    decomposition_data = tsne.fit_transform(weight)
    x = []
    y = []
    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])

    left = random.randint(0, (len(x) // num) - 1) * num
    right = left + num

    # scatter plot
    plt.scatter(x[left:right], y[left:right], c=k_means.labels_[left:right], marker=".")
    # Remove the labels on the axes
    plt.xticks(())
    plt.yticks(())
    for i in range(left, right):
        # add an annotation to each point, which is the key word of each response
        # xytext is used for the coordinate of the label
        plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(x[i] + 0.1, y[i] + 0.1))
    plt.show()


def generate_result_file(csv_file: str, clusters: list, key_words: list,
                         corpus: list, preprocessed_corpus: list) -> None:
    """Generate a csv file to store the result of clustering.
    The first column is the cluster number of each response.
    The second column is the key word of each response.
    The third column is the original text of each response.
    The fourth column is the preprocessed text of each response.
    """
    info = {'cluster': [], 'key word': [], 'response': [], 'processed response': []}
    for i in range(0, len(clusters)):
        for j in range(0, len(clusters[i])):
            info['cluster'].append(i)

            idx = clusters[i][j]
            info['key word'].append(key_words[idx])
            info['response'].append(corpus[idx])
            info['processed response'].append(preprocessed_corpus[idx])
    # create a dataframe using info
    df = pd.DataFrame(info)
    df.to_csv(csv_file, index=False)


def clustering(data_file: str, result_file: str, col: int, n: int, point_nums: int = 100) -> None:
    """Do clustering using tf-idf and k-means.
    Note:
        txt_file is a text file that stores the responses.
        csv_file is a csv file that will store the result of clustering. (For more information,
        refer to the document of function generate_result_file)
        n is the number of clusters.
    """
    csv_to_txt(data_file, 'temp.txt', col)
    corpus, preprocessed_corpus = generate_corpus('temp.txt')
    weight, key_words = tf_idf(preprocessed_corpus)
    kmeans, clusters = k_means(n, weight)
    generate_graph(weight, kmeans, key_words, point_nums)
    generate_result_file(result_file, clusters, key_words, corpus, preprocessed_corpus)
    os.remove('temp.txt')


if __name__ == '__main__':
    # generate a text file storing the responses in column 8 of coded_data.csv
    clustering('coded_data.csv', 'col_18_result1.csv', col=18, n=20)
