"""Calculates the output value from RIPA equation"""
# To use the file, scroll to line 106 and input words you'd like to use

import math

LAMBDA = 1
ALPHA = -1


def ripa(data_file: str, vec_file: str, voca_file: str, window_size: int,
         w: str, x: str, y: str) -> None:
    """RIPA equation function"""
    corpus_size = get_corpus_size(data_file)

    pmi = math.log(get_joint_p(corpus_size, x, y, window_size, data_file) /
                   (get_p(corpus_size, x, voca_file) * get_p(corpus_size, y, voca_file)))
    cspmi = pmi + math.log(get_joint_p(corpus_size, x, y, window_size, data_file))
    c_value = (1 / (math.sqrt(LAMBDA))) / math.sqrt((-1 * cspmi + ALPHA))

    prob_x_w = get_joint_p(corpus_size, x, w, window_size, data_file)
    prob_y_w = get_joint_p(corpus_size, y, w, window_size, data_file)

    z_x = get_z_term(x, vec_file)
    z_y = get_z_term(y, vec_file)

    ripa = c_value * (math.log(prob_x_w / prob_y_w) - z_x + z_y)

    print('ripa(' + w + ', ' + x + ', ' + y + ') = ' + str(ripa))


def get_z_term(word: str, vec_file: str) -> float:
    """Return bias value z of given word"""
    f = open(vec_file, "r")
    for row in f:
        row_lst = row.split()
        if row_lst[0] == word:
            return float(row_lst[-1])


def get_joint_p(corpus_size, x: str, y: str, window_size: int, data_file: str) -> float:
    """Return joint probability of x and y:
       number of times x is followed by y in a window of max w words"""
    joint_occurrence = 0
    f = open(data_file, "r")
    for row in f:
        x_indices = get_all_index(row, x)
        y_indices = get_all_index(row, y)
        for x_idx in x_indices:
            for y_idx in y_indices:
                if 0 <= y_idx - x_idx <= window_size:
                    joint_occurrence += 1

    # i am eating lunch because my family are eating lunch, repeated occurrence?
    # x = eating
    # y = lunch
    # x_indices = 2, 8
    # y_indices = 3, 9
    # or number of times they occur in the same doc
    return joint_occurrence / corpus_size


def get_p(corpus_size, word: str, voca_file: str) -> float:
    """Return probability of given word"""
    occurrence = 0
    f = open(voca_file, "r")
    for row in f:
        row_lst = row.split()
        if row_lst[0] == word:
            occurrence = int(row_lst[1])
    return occurrence / corpus_size


def get_all_index(sentence: str, word: str) -> list:
    """Return all indices of this word in this sentence"""
    sentence = sentence.split()
    lst = []
    for idx in range(0, len(sentence)):
        if sentence[idx] == word:
            lst.append(idx)
    return lst


def get_corpus_size(data_file: str) -> int:
    """Return corpus size (total num of words in file)"""
    size = 0
    f = open(data_file, "r")
    for row in f:
        size += len(row.split())
    return size


if __name__ == '__main__':
    # x = 'word', y = 'word', w = 'word', change into your own choices
    # run it, prints 3 numbers, first one is p(x, w), second one is p(y, w), last one is RIPA value
    # if outputs math domain error, means joint probability of one word with attribute word is 0,
    # that they have never co-occured, hence log doesn't apply, RIPA can't be calculated
    ripa(data_file='result/survey_data.txt', vec_file='result/vectors.txt',
         voca_file='result/vocab.txt', window_size=10,
         x='person', y='onlin', w='hard')
