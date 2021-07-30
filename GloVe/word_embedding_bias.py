"""Calculate g(w, x, y)"""
from typing import Tuple


def g(vector_file: str, w: str, x: str, y: str) -> None:
    """Print the value of g(w, x, y).
    Note:
        vector_file is the txt file of vectors generated from GloVe.
    """
    vec_w, vec_x, vec_y = get_vectors(vector_file, w, x, y)
    g = calculate_g(vec_w, vec_x, vec_y)
    print('g(' + w + ', ' + x + ', ' + y + ') = ' + str(g))


def calculate_g(w: list, x: list, y: list) -> float:
    """g(w, x, y) = w·(x-y) / ||x-y||
    preconditions:
        - len(w) == len(x)
        - len(x) == len(y)
    """
    sum_of_product_so_far = 0
    sum_of_square_so_far = 0

    for i in range(len(w)):
        sum_of_product_so_far += w[i] * (x[i] - y[i])
        sum_of_square_so_far += (x[i] - y[i]) ** 2

    return sum_of_product_so_far / (sum_of_square_so_far ** 0.5)


def get_vectors(txt_file: str, word_w: str, word_x: str, word_y: str) \
        -> Tuple[list, list, list]:
    """Return a tuple of three lists. Each list represents a vector for each word."""

    vec_w = []
    vec_x = []
    vec_y = []

    file = open(txt_file, 'r')
    lines = file.readlines()

    for line in lines:
        words = line.split(' ')
        if word_w == words[0]:
            vec_w = words[1:-1]
            vec_w = [float(i) for i in vec_w]
        if word_x == words[0]:
            vec_x = words[1:-1]
            vec_x = [float(i) for i in vec_x]
        if word_y == words[0]:
            vec_y = words[1:-1]
            vec_y = [float(i) for i in vec_y]

    return (vec_w, vec_x, vec_y)


if __name__ == '__main__':
    g(vector_file='result/vectors.txt', w='hard', x='person', y='onlin')
