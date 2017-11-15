#!/usr/bin/python3
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Roshith Raghavan, Vaibhav Shah
# (based on skeleton code by D. Crandall, Oct 2017)
#
#####
#
#
# Here we used the eBook corpus provided under the Gutenberg Project, called the "Coming of Man".
# This is the corpus we used to get the transition probabilities of a character to another character. We also calculated
# the initial probabilities of a character from this corpus. We are uploading the corpus along with our code on Github.
# For emission probabilities, we used the training image and training string provided by D. Crandall in the skeleton
# code. Our emission probabilities are calculated as the number of spaces/ asterisks that are located at the same place
# in the tags and the image, divided by the total number of pixels, 25 * 14 = 250.
#
# When we initially ran our code, one problem we faced was the priors and transition probabilities over shadowed our
# emissions. Upon checking we realized that our emissions would give more accurate results on its own.(Naive Bayes).
# But because the probabilities for the different tags were so close, that state transitions and priors would overshadow
# the emissions.
#
# To correct for this, we have penalized emissions by exponentiating them to 250. This allows for huge differences post
# exponentiation, even if the differences were small before. This significantly improved our results.
#
# We also spoke with the AI, Archana. We explained this to her and asked for her opinion. She suggested smoothing and
# explained a basic smoothing operation. She also asked us to research Laplace smoothing. We tried both smoothing
# appraoches, but they decreased the performance of our results.
#
# The accuracies that we got are:
# Accuracy:
# Naive Bayes: 			 0.5639246778989098
# Variable Elimination:  0.5599603567888999
# Viterbi: 				 0.599603567888999
#
# We are only 2 people in the team and unfortunately both of us have assignment submissions this week. We had planned
# out the last 2 weeks and accordingly we have to give priority to our other assignments for the remainder of the week.
# As such we are meeting the original deadline, hoping that due consideration will be given. Even Archana sugested that
# we submit it by the original deadline.
#
#
#####

import heapq
import operator

from PIL import Image, ImageDraw, ImageFont
import sys
import time
import math

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    # print(im.size)
    # print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [["".join(['*' if px[x, y] < 1 else ' '
                             for x in range(x_beg, x_beg + CHARACTER_WIDTH)])
                    for y in range(0, CHARACTER_HEIGHT)], ]
    return result


def load_training_letters(fname):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}


def train(fname):
    global state_transitions
    global initial_tags
    global initial_state_distribution
    global tag_dict
    global priors
    with open(fname, 'r') as file:
        list_of_valid_char = set()
        list_of_invalid_char = set()
        for line in file:
            for index, character in enumerate(str(line)):
                if str(character) in tags:
                    list_of_valid_char.add(character)
                    tag_dict[character] += 1
                    tag_dict["total"] += 1

                    if index == 0:
                        if character in tags:
                            initial_tags[character] += 1
                            initial_tags["total"] += 1
                    if index > 0:
                        if (str(line)[index - 1]) in tags:
                            try:
                                state_transitions[str(character)][(str(line)[index - 1])] += 1
                                state_transitions[str(character)]['total'] += 1
                            except KeyError:
                                print(KeyError)
                                pass
                else:
                    list_of_invalid_char.add(character)

    for key in tags:
        priors[key] = tag_dict[key] / tag_dict["total"]
        initial_state_distribution[key] = initial_tags[key] / initial_tags["total"]



def simplified(image):
    final_pos = []
    for character in load_letters(image):
        pos = {}
        word_dict = char_prob(character)
        for tag in tags:
            pos[tag] = 100000
        for key in tags:

            if priors[key]:
                if word_dict[key]:
                    P = - math.log(word_dict[key]) \
                        - math.log(priors[key])
                else:
                    P = -math.log(undef_prob)
            else:
                if word_dict[key]:
                    P = - math.log(word_dict[key]) \
                        - math.log(undef_prob)
                else:
                    P = -math.log(undef_prob)

            pos[key] = P
        final_pos.append(min(pos.items(), key=operator.itemgetter(1))[0])

    return final_pos


def hmm_ve(image):
    ans = []
    tau = 0.0
    prev_tag = None
    count = 0
    for character in load_letters(image):
        count += 1
        word_dict = char_prob(character)
        max_tau = 100000000
        max_prev_tag = " "

        for key in tags:
            if not prev_tag:
                if word_dict[key]:
                    prob = - math.log(word_dict[key])
                else:
                    prob = - math.log(undef_prob)

                if prob < max_tau:
                    max_tau = prob
                    max_prev_tag = key

            else:
                if state_transitions[key][prev_tag]:

                    if word_dict[key]:
                        prob = - (math.log(state_transitions[key][prev_tag] /
                                           state_transitions[key]["total"])) \
                               - math.log(word_dict[key]) + tau
                    else:
                        prob = - (math.log(state_transitions[key][prev_tag] /
                                           state_transitions[key]["total"])) \
                               - math.log(undef_prob) + tau

                else:
                    if word_dict[key]:
                        prob = - (math.log(undef_prob)) \
                               - math.log(word_dict[key]) + tau

                if prob < max_tau:
                    max_tau = prob
                    max_prev_tag = key

        tau = max_tau
        prev_tag = max_prev_tag
        ans.append(prev_tag)
        if count == -1:
            break

    return ans


def hmm_viterbi(image):
    sentence = load_letters(image)
    word_dict = char_prob(sentence[0])
    for key in tags:

        if word_dict[key]:
            adjacency_list['0'][key] = (-math.log(word_dict[key]), 'Source')
        else:
            adjacency_list['0'][key] = (-math.log(undef_prob), 'Source')

    for i in range(1, len(sentence)):
        adjacency_list[str(i)] = {}
        word_dict = char_prob(sentence[i])
        for key in tags:
            adj_list = []
            for k in tags:

                if not state_transitions[key][k]:
                    if word_dict[key]:
                        adj_list.append((adjacency_list[str(i - 1)][k][0] -
                                         (math.log(undef_prob)) +
                                         -math.log(word_dict[key]), k, sentence[i]))
                    else:
                        adj_list.append((adjacency_list[str(i - 1)][k][0] -
                                         (math.log(undef_prob)) +
                                         -math.log(undef_prob), k, sentence[i]))

                else:
                    if word_dict[key]:
                        adj_list.append((adjacency_list[str(i - 1)][k][0] -
                                         (math.log(state_transitions[key][k] /
                                                   state_transitions[key]['total'])) +
                                         -math.log(word_dict[key]), k, sentence[i]))
                    else:
                        adj_list.append((adjacency_list[str(i - 1)][k][0] -
                                         (math.log(state_transitions[key][k] /
                                                   state_transitions[key]['total'])) +
                                         -math.log(undef_prob), k, sentence[i]))

            heapq.heapify(adj_list)
            adjacency_list[str(i)][key] = heapq.heappop(adj_list)
    PoS = []
    final_PoS = []
    for i in range(0, len(sentence), 1):
        if i == 0:
            PoS.append(min(adjacency_list[str(len(sentence) - 1 - i)].items(), key=operator.itemgetter(1)))
        else:
            PoS.append((PoS[-1][1][1], adjacency_list[str(len(sentence) - 1 - i)][PoS[-1][1][1]]))
    for values in PoS:
        final_PoS.append(values[0])

    return final_PoS[::-1]


def print_letter(letter):
    print("\n".join([r for r in letter]))


def print_string(letter):
    try:
        return ("".join([r for r in letter]))
    except TypeError:
        return " " * len(letter)


def similarity(char, tag):
    hit = 0
    for i in range(CHARACTER_HEIGHT):
        for j in range(CHARACTER_WIDTH):
            if tag[i][j] == char[i][j]:
                hit += 1
    return math.pow(hit / (CHARACTER_WIDTH * CHARACTER_HEIGHT), 250)


def char_prob(char):
    prob = {}
    for tag in tags:
        sim = similarity(char, train_letters[tag])
        sim /= (tag_dict[tag] + tag_dict["total"])
        prob[tag] = sim
    return prob


def calculate_accuracy(gt, nb, ve, vit):
    gt = gt.replace("\n", "")
    gt = gt.rstrip()
    for i in range(len(gt)):
        if gt[i] == gt[i]:
            accuracy["gt"] += 1
        if gt[i] == nb[i]:
            accuracy["nb"] += 1
        if gt[i] == ve[i]:
            accuracy["ve"] += 1
        if gt[i] == vit[i]:
            accuracy["vit"] += 1


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
zero = 0
tag_dict = {"total": zero}
state_transitions = {}
initial_tags = {"total": zero}
initial_state_distribution = {}
priors = {}
adjacency_list = {'0': {}}
for tag_variable in tags:

    tag_dict[tag_variable] = zero

    initial_tags[tag_variable] = zero
    initial_state_distribution[tag_variable] = zero

    state_transitions[tag_variable] = {'total': zero}
    for t in tags:
        state_transitions[tag_variable][t] = zero

train(train_txt_fname)
print(state_transitions)
accuracy = {"gt": 0, "nb": 0, "ve": 0, "vit": 0}
file = open("test-strings.txt", "r")
lines = file.readlines()
for i in range(20):
    image = "test-" + str(i) + "-0.png"
    test_letters = load_letters(image)

    gt = lines[i]
    gt = gt.replace("\n", "")
    nb = simplified(image)
    ve = hmm_ve(image)
    vit = hmm_viterbi(image)
    calculate_accuracy(gt, nb, ve, vit)

    print("Ground Truth: \t\t\t\t\t\t\t", gt)
    print("Naive Bayes: \t\t\t\t\t\t\t", print_string(nb))
    print("Variable Elimination: \t\t\t\t\t", print_string(ve))
    print("Viterbi: \t\t\t\t\t\t\t\t", print_string(vit))
    print("\n\n\n")

print("Accuracy:")
print("Naive Bayes: \t\t\t", accuracy["nb"] / accuracy["gt"])
print("Variable Elimination: \t", accuracy["ve"] / accuracy["gt"])
print("Viterbi: \t\t\t\t", accuracy["vit"] / accuracy["gt"])
print("Running Time: ", time.time() - start)
