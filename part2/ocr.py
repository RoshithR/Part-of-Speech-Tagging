#!/usr/bin/python3
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2017)
#
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
                             for x in range(x_beg, x_beg+CHARACTER_WIDTH)])
                    for y in range(0, CHARACTER_HEIGHT)], ]
    return result


def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}


def train(fname):
    global state_transitions
    global initial_tags
    global initial_state_distribution
    global tag_dict
    global priors
    with open(fname, 'r') as file:
        print("encoding", file.encoding)
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
                        if (str(line)[index-1]) in tags:
                            try:
                                state_transitions[str(character)][(str(line)[index-1])] += 1
                                state_transitions[str(character)]['total'] += 1
                            except KeyError:
                                print(KeyError)
                                pass
                else:
                    list_of_invalid_char.add(character)

    for key in tags:
        priors[key] = tag_dict[key] / tag_dict["total"]
        initial_state_distribution[key] = initial_tags[key] / initial_tags["total"]
    print(set(tags) - list_of_valid_char)
    print(list_of_invalid_char)


def simplified(image):
    final_pos = []
    for character in load_letters(image):
        # print_letter(character)
        pos = {}
        word_dict = char_prob(character)
        # print(word_dict)
        for tag in tags:
            pos[tag] = 100000
        for key in tags:
            P = -math.log(word_dict[key])

            # P = - math.log(word_dict[key]) \
            #     - math.log(priors[key])

            # print("key: ", key)
            # print(word_dict[key])
            # print(priors[key])
            # print("\n\n\n")
            # if str(character) in char_prob(key):
            #     P = - math.log(char_prob(key)[str(character)]) \
            #         - math.log(priors[key])
            # else:
            #     P = -math.log(undef_prob)
            pos[key] = P
        # print(pos)
        final_pos.append(min(pos.items(), key=operator.itemgetter(1))[0])
        # break
    # print(final_pos)
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
        # print_letter(character)

        for key in tags:
            if first:
                # prob = -math.log(initial_state_distribution[key] *
                #                  word_dict[key])

                prob = - math.log(word_dict[key])
                # print("key: ", key)
                # print("Word_dict: ", word_dict[key])
                # print


                # if character in word_dict[key]:
                #     prob = -math.log(initial_state_distribution[key] *
                #                      word_dict[key])
                # else:
                #     prob = -math.log(initial_state_distribution[key]) - math.log(undef_prob)

                if prob < max_tau:
                    max_tau = prob
                    max_prev_tag = key

            else:
                if state_transitions[key][prev_tag] >= 1:
                    # print("key: ", key, "\tprev_tag: ", prev_tag)
                    # print("State_transition count: ", state_transitions[key][prev_tag])
                    # print("State_transition total: ", state_transitions[key]["total"])
                    # print("Word_dict: ", word_dict[key])
                    # print("Tau: ", tau)
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


def hmm_viterbi(image):
    # PoS = []
    sentence = load_letters(image)
    word_dict = char_prob(sentence[0])
    for key in tags:

        adjacency_list['0'][key] = (-math.log(word_dict[key]), 'Source')

        # adjacency_list['0'][key] = (-math.log(initial_state_distribution[key]) +
        #                             word_dict[key], 'Source')

        # if sentence[0] in word_dict[key]:
        #     adjacency_list['0'][key] = (-math.log(initial_state_distribution[key]) +
        #                                 word_dict[key], 'Source')
        # else:
        #     adjacency_list['0'][key] = (
        #         -math.log(initial_state_distribution[key]) - math.log(undef_prob), 'Source')

    for i in range(1, len(sentence)):
        adjacency_list[str(i)] = {}
        word_dict = char_prob(sentence[i])
        for key in tags:
            adj_list = []
            for k in tags:

                # if state_transitions[key][k] <= undef_prob:
                #     state_transitions[key][k] = undef_prob
                adj_list.append((adjacency_list[str(i - 1)][k][0] -
                                 math.log(state_transitions[key][k] /
                                          state_transitions[key]['total']) +
                                 -math.log(word_dict[key]), k, sentence[i]))

                # if sentence[i] in word_dict[key]:
                #     if k not in state_transitions[key]:
                #         state_transitions[key][k] = undef_prob
                #     adj_list.append((adjacency_list[str(i - 1)][k][0] -
                #                      math.log(state_transitions[key][k] /
                #                               state_transitions[key]['total']) -
                #                      math.log(word_dict[key][str(sentence[i])] /
                #                               tag_dict[key]), k, sentence[i]))
                # else:
                #     if k not in state_transitions[key]:
                #         state_transitions[key][k] = undef_prob
                #     adj_list.append((adjacency_list[str(i - 1)][k][0] -
                #                      math.log(state_transitions[key][k] /
                #                               state_transitions[key]['total']) -
                #                      math.log(undef_prob), k, sentence[i]))
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
    # print(final_PoS)
    return final_PoS[::-1]


def print_letter(letter):
    print("\n".join([r for r in letter]))


def print_string(letter):
    print("".join([r for r in letter]))


def similarity(char, tag):
    hit = 0
    for i in range(CHARACTER_HEIGHT):
        for j in range(CHARACTER_WIDTH):
            if tag[i][j] == char[i][j]:
                hit += 1
    return hit / (CHARACTER_WIDTH * CHARACTER_HEIGHT)


def char_prob(char):
    prob = {}
    total = 0
    for tag in tags:
        sim = similarity(char, train_letters[tag])
        prob[tag] = sim
    #     try:
    #         total += math.exp(sim)
    #     except OverflowError:
    #         print(total)
    #         print(sim)
    #         print(math.exp(sim))
    #         exit(69)
    # # total = math.log(total)
    # # for tag in tags:
    # #     prob[tag] -= total
    # prob = sorted(prob.items(), key=operator.itemgetter(1))
    return prob


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
tag_dict = {"total": undef_prob}
state_transitions = {}
initial_tags = {"total": undef_prob}
initial_state_distribution = {}
priors = {}
adjacency_list = {'0': {}}
for tag_variable in tags:

    tag_dict[tag_variable] = undef_prob

    initial_tags[tag_variable] = undef_prob
    initial_state_distribution[tag_variable] = undef_prob

    state_transitions[tag_variable] = {'total': undef_prob}
    for t in tags:
        state_transitions[tag_variable][t] = undef_prob

train(train_txt_fname)
# print(state_transitions)
# exit(69)
# test_img_fname = train_img_fname
print("\n\nNaive Bayes")
print_string(simplified(test_img_fname))
print("\n\nVariable Elimination")
print_string(hmm_ve(test_img_fname))
print("\n\nViterbi")
print_string(hmm_viterbi(test_img_fname))


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
# print_letter(test_letters[0])
# print(char_prob(test_letters[0]))
# print(len(test_letters[0]))


# print(test_letters)
# Same with test letters. Here's what the third letter of the test data
#  looks like:
# for i in range(len(test_letters)):
#     print_letter(test_letters[i])
# print(test_letters)

print("Running Time: ", time.time() - start)
