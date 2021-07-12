""" Helper functions for reading 
"""

import numpy as np

# helperfunctions


def where_in(word, lines):
    """ Finds where word is in lines and returns array
        :param word: word/s to look for
        :type word: str/list for multiple words
        :param lines: lines to scan
        :type lines: list(str)
        :return where_array: array with the indices
        :rtype: numpy array
    """
    if isinstance(word, str):
        word = [word]

    where_array = np.where(
        np.array([all(word_i in line for word_i in word) for line in lines], dtype=int) == 1)[0]

    return where_array

def where_in_any(word, lines):
    """ Finds where word is in lines and returns array
        :param word: word/s to look for
        :type word: str/list for multiple words
        :param lines: lines to scan
        :type lines: list(str)
        :return where_array: array with the indices
        :rtype: numpy array
    """
    if isinstance(word, str):
        word = [word]

    where_array = np.where(
        np.array([any(word_i in line for word_i in word) for line in lines], dtype=int) == 1)[0]

    return where_array

def where_is(word, lines):
    """ Finds where line corresponds to the required word
        :param word: word to look for
        :type word: str
        :param lines: lines to scan
        :type lines: list(str)
        :return where_array: array with the indices
        :rtype: numpy array
    """

    where_array = np.where(
        np.array([line == word for line in lines], dtype=int) == 1)[0]

    return where_array
