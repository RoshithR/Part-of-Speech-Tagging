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
import math
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

def train():
    global state_transitions
    global initial_tags
    global initial_state_distribution
    global tag_dict
    global priors
    with open('Iliad.txt', 'r') as file:
        for line in file:
            for index,character in enumerate(str(line)):
                if str(character) in tags:
                    tag_dict[character]+=1
                    tag_dict["total"] += 1

                    if index==0:
                        if (character) in tags:
                            initial_tags[character]+=1
                            initial_tags["total"]+=1
                    if index>0:
                        if (str(line)[index-1]) in tags:
                            try:
                                state_transitions[str(character)][(str(line)[index-1])]+=1
                                state_transitions[str(character)]['total']+=1
                            except KeyError:
                                pass
    for key in tags:
        priors[key] = tag_dict[key] / tag_dict["total"]
        initial_state_distribution[key]=initial_tags[key] / initial_tags["total"]

def simplified(image):
    final_pos = []
    for character in load_letters(image):
        print_letter(character)
        pos = {}
        word_dict =char_prob(character)
        # print(word_dict)
        for tag in tags:
            pos[tag] = 100000
        for key in tags:

            # print(key)
            # P = - math.log(word_dict[key]) \
            #     - math.log(priors[key])
            P = - math.log(word_dict[key])
            # print("key: ", key)
            # print(word_dict[key])
            # print(priors[key])
            # print("\n\n\n")
            # if str(character) in char_prob(key):
            #     P = - math.log(char_prob(key)[str(character)]) \
            #         - math.log(priors[key])
            # else:
            #     P = -math.log(0.0000000001)
            pos[key] = P
        # print(pos)
        final_pos.append(min(pos.items(), key=operator.itemgetter(1))[0])
        # break
    print(final_pos)
    return final_pos


def hmm_ve(image):
    ans = []
    first = True
    tau = 0.0
    prev_tag = None
    for character in load_letters(image):
        word_dict = char_prob(character)
        max_tau = 1000
        max_prev_tag = None

        for key in tags:
            if first:
                # prob = -math.log(initial_state_distribution[key] *
                #                  word_dict[key])

                prob = -math.log(word_dict[key])



                # if character in word_dict[key]:
                #     prob = -math.log(initial_state_distribution[key] *
                #                      word_dict[key])
                # else:
                #     prob = -math.log(initial_state_distribution[key]) - math.log(undef_prob)

                if prob < max_tau:
                    max_tau = prob
                    max_prev_tag = key

            else:
                if prev_tag in state_transitions[key]:
                    prob = - math.log(state_transitions[key][prev_tag] /
                                      state_transitions[key]["total"]) \
                           - math.log(word_dict[key]) + tau
                else:
                    prob = - math.log(undef_prob) \
                           - math.log(word_dict[key]) + tau

                # if character in word_dict[key]:
                #     if prev_tag in state_transitions[key]:
                #         prob = - math.log(state_transitions[key][prev_tag] /
                #                           state_transitions[key]["total"]) \
                #                - math.log(word_dict[key]) + tau
                #     else:
                #         prob = - math.log(undef_prob) \
                #                - math.log(word_dict[key]) + tau
                # else:
                #     if prev_tag in state_transitions[key]:
                #         prob = - math.log(state_transitions[key][prev_tag] /
                #                           state_transitions[key]["total"]) \
                #                - math.log(undef_prob) + tau
                #     else:
                #         prob = - math.log(undef_prob) \
                #                - math.log(undef_prob) + tau

                if prob < max_tau:
                    max_tau = prob
                    max_prev_tag = key

        tau = max_tau
        prev_tag = max_prev_tag
        ans.append(prev_tag)

        first = False
    return ans


def print_letter(letter):
    print("\n".join([r for r in letter]))


def similarity(char, tag):
    hit = 0
    for i in range(len(char)):
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

    # prob = sorted(prob.items(), key=operator.itemgetter(1))
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


undef_prob = 0.0000000001
tag_dict = {"total": 0.0000000001}
state_transitions={}
initial_tags={"total":0.0000000001}
initial_state_distribution={}
priors = {}
for tag in tags:

    tag_dict[tag] = 0.0000000001


    initial_tags[tag]=0.0000000001
    initial_state_distribution[tag]=0.0000000001

    state_transitions[tag]={'total':0.0000000001}
    for t in tags:
        state_transitions[tag][t]=0.0000000001




train()
print(simplified(test_img_fname))

print(hmm_ve(test_img_fname))
# print(state_transitions)
# print(initial_tags)
# print(initial_state_distribution)


# PS1, PSip1Si, PWiSi, PCount = learn_hmm(train_txt_fname)

# Below is just some sample code to show you how the functions above work.
# You can delete them and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
# print_letter(train_letters['a'])
# print(char_prob(test_letters[0]))
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
