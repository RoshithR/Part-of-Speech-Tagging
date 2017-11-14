#!/usr/bin/python3
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2017)
#
import operator

from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import sys
import time
import pickle
from pathlib import Path

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25


def pickle_data_save(fname, t_PS1, t_PSip1Si, t_PWiSi, t_PCount):
    with open(fname, "wb") as f:
        pickle.dump(t_PS1, f)
        pickle.dump(t_PSip1Si, f)
        pickle.dump(t_PWiSi, f)
        pickle.dump(t_PCount, f)


def pickle_data_load(fname):
    with open(fname, "rb") as f:
        t_PS1 = pickle.load(f)
        t_PSip1Si = pickle.load(f)
        t_PWiSi = pickle.load(f)
        t_PCount = pickle.load(f)
        return t_PS1, t_PSip1Si, t_PWiSi, t_PCount


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [["".join(['*' if px[x, y] < 1 else ' '
                             for x in range(x_beg, x_beg+CHARACTER_WIDTH)])
                    for y in range(0, CHARACTER_HEIGHT)], ]
    return result


def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}


def print_letter(letter):
    print("\n".join([r for r in letter]))


def similarity(char, tag):
    hit = 0
    print(len(char))
    print(tag)
    for i in range(len(char)):
        print(i)
        if tag[i] == char[i]:
            hit += 1
    return hit / len(char)


def char_prob(char):
    prob = {}
    total = 0
    for tag in tags:
        sim = similarity(char, train_letters[tag])
        prob[tag] = sim
        total += sim

    for tag in tags:
        prob[tag] /= total

    prob = sorted(prob.items(), key=operator.itemgetter(1))
    return prob


def learn_hmm(fname):
    t_PS1 = {}
    t_PSip1Si = pd.DataFrame(data=0.0, index=tags, columns=tags, dtype=float)
    t_PWiSi = {}
    t_PCount = {}

    file = open(fname, 'r')
    for line in file:
        data = tuple([w.lower() for w in line.split()])
        Sentence = " ".join([r for r in data[0::2]])
        print(Sentence)

        # it = iter(data)
        #
        # word = next(it)
        # tag = next(it)
        # # print(word, " \t", tag)
        #
        # # Calculating Probabilities of P(S1)
        # if tag in t_PS1:
        #     t_PS1[tag] += 1
        # else:
        #     t_PS1[tag] = 1
        #
        # # Updating word count for Emission Probabilities for first word of each sentence.
        # if tag in t_PWiSi:
        #     if word in t_PWiSi[tag]:
        #         t_PWiSi[tag][word] += 1
        #     else:
        #         t_PWiSi[tag][word] = 1
        # else:
        #     t_PWiSi[tag] = {}
        #     if word in t_PWiSi[tag]:
        #         t_PWiSi[tag][word] += 1
        #     else:
        #         t_PWiSi[tag][word] = 1
        #
        # # Updating tag count for Emission Probabilities for first tag of each word.
        # if tag in t_PCount:
        #     t_PCount[tag] += 1
        # else:
        #     t_PCount[tag] = 1
        #
        # for word in it:
        #     prev_tag = tag
        #     tag = next(it)
        #
        #     t_PSip1Si.loc[prev_tag, tag] += 1
        #
        #     # Updating word count for Emission Probabilities for word of each sentence.
        #     if tag in t_PWiSi:
        #         if word in t_PWiSi[tag]:
        #             t_PWiSi[tag][word] += 1
        #         else:
        #             t_PWiSi[tag][word] = 1
        #     else:
        #         t_PWiSi[tag] = {}
        #         if word in t_PWiSi[tag]:
        #             t_PWiSi[tag][word] += 1
        #         else:
        #             t_PWiSi[tag][word] = 1
        #
        #     # Updating tag count for Emission Probabilities for tag of each word.
        #     if tag in t_PCount:
        #         t_PCount[tag] += 1
        #     else:
        #         t_PCount[tag] = 1
        break

    return t_PS1, t_PSip1Si, t_PWiSi, t_PCount


#####
# main program
start = time.time()
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

train_string = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
tags = [char for char in train_string]
print(tags)


# PS1, PSip1Si, PWiSi, PCount = learn_hmm(train_txt_fname)

# Below is just some sample code to show you how the functions above work.
# You can delete them and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
# print_letter(train_letters['a'])
print(char_prob(test_letters[0]))
# print(len(test_letters[0]))


# print(test_letters)
# Same with test letters. Here's what the third letter of the test data
#  looks like:
# for i in range(len(test_letters)):
#     print_letter(test_letters[i])
# print(test_letters)

# pickle_file = "hmm_probabilities.p"
# my_file = Path(pickle_file)
# if my_file.is_file():
#     print("Pickled File Exists")
#     PS1, PSip1Si, PWiSi, PCount = pickle_data_load(pickle_file)
# else:
#     print("Pickled File Does Not Exists")
#     PS1, PSip1Si, PWiSi, PCount = learn_hmm(train_txt_fname)
#     pickle_data_save(pickle_file, PS1, PSip1Si, PWiSi, PCount)


# print(PS1)
# print(sum(PS1.values()))
# print(PSip1Si)
# print(PWiSi)
# print(PCount)
print("Running Time: ", time.time() - start)
