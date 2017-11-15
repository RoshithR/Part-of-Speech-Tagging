###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids:
#   Roshith Raghvan- roragh
#   Vaibhav Shah- vaivshah
#
# (Based on skeleton code by D. Crandall)
#
#
####
# We used the Brown corpus training data to train our models. From these the transitions probabilities, emission
# probabilities and initial probability distribution were computed and stored to dictionaries.
# When a word is found in test file but not in training file it is given a default probability of 0.0000000000001
# We are converting the probabilities to negative log and then minimizing cost instead of maximizing probabilities.
# In variable elimination, we traverse from the nodes with maximum probability. When traversing we take the product of
# the posterior of the node along with its transition to a node in the next interval in time.
# In viterbi, we are using graph traversal from the most likely final word tag and traversing to preceding nodes in the
# chain. Cost minimization in viterbi is accomplished through min heap pop operation.
# For viterbi, Triple nested dictionary is used to represent the graph.
#
# Word tagging Accuracy when measured against bc.test:
# Simplified: 92.10%  HMM VE: 91.79%  HMM MAP: 95.18%
#
# Sentence accuracy with three inference approaches when measured against bc,test:
# Simplified: 39.25%  HMM VE: 38.90%   HMM MAP: 55.30%?
####
import heapq
import math
import operator


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    def __init__(self):
        # Setting up the tags
        self.tags = ["adj", "adv", "adp", "conj", "det", "noun", "num", "pron", "prt", "verb", "x", "."]
        self.undef_prob = 0.00000000001
        self.tag_dict = {"total": self.undef_prob}
        self.word_dict ={}
        self.initial_tags = {"total": self.undef_prob}
        self.initial_state_distribution = {}
        self.state_transitions = {}

        for tag in self.tags:
            # tag_dict maintains count of occurrences of each tag and the total number of tags
            self.tag_dict[tag] = self.undef_prob

            # word_dict maintains occurrence of each word within a specific tag
            self.word_dict[tag] = {}
            self.word_dict[tag]["total"] = self.undef_prob

            # intial_tags stores the number of times a part of speech tag occurs at the start of a sentence
            self.initial_tags[tag] = self.undef_prob

            # initial probabilities are computed and stored here as initial_tag=def/total
            # number of initial tags from above dictionary
            self.initial_state_distribution[tag] = self.undef_prob

            # number of times a given tag(the key here) is preceded by another PoS tag. To compute P(X2=Det|X1=Noun)
            self.state_transitions[tag] = {}
            self.state_transitions[tag]["total"] = self.undef_prob

        # priors computes the number of time a tag occurs in testing data i.e. #adj/total
        self.priors = {}

        # Graph for viterbi algorithm
        self.adjacency_list = {'0': {}}

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling if str(word) in self.word_dict["adj"]:

    def posterior(self, sentence, label):
        total = 0.0
        prev_tag = None
        prob = 0.0
        for word, tag in zip(sentence, label):

            if not prev_tag:
                if word in self.word_dict[tag]:
                    prob += -math.log(self.initial_state_distribution[tag] *
                                      self.word_dict[tag][word] / self.word_dict[tag]["total"])
                else:
                    prob += -math.log(self.initial_state_distribution[tag] *
                                      self.undef_prob)

                prev_tag = tag
            else:
                if word in self.word_dict[tag]:
                    prob += -math.log(self.state_transitions[tag][prev_tag] /
                                      self.state_transitions[tag]["total"] *
                                      self.word_dict[tag][word] / self.word_dict[tag]["total"])
                else:
                    prob += -math.log(self.state_transitions[tag][prev_tag] /
                                      self.state_transitions[tag]["total"] *
                                      0.0000000001)
                prev_tag = tag
            total += prob
        return -prob

    # Do the training!
    #
    def train(self, data):

        for line in data:
            self.initial_tags[line[1][0]] += 1
            self.initial_tags["total"] += 1
            for word, tag in zip(line[0], line[1]):
                self.tag_dict[tag] += 1
                self.tag_dict["total"] += 1

                if word in self.word_dict[tag]:
                    self.word_dict[tag][word] += 1
                    self.word_dict[tag]["total"] += 1
                else:
                    self.word_dict[tag][word] = 1
                    self.word_dict[tag]["total"] += 1

            # training transition probabilities
            for index, tag in enumerate(line[1]):
                if index > 0:
                    if line[1][index-1] in self.state_transitions[tag]:
                        self.state_transitions[tag][line[1][index-1]] += 1
                        self.state_transitions[tag]["total"] += 1
                    else:
                        self.state_transitions[tag][line[1][index - 1]] = 1
                        self.state_transitions[tag]["total"] += 1

        for key in self.tags:
            self.priors[key] = self.tag_dict[key]/self.tag_dict["total"]
            self.initial_state_distribution[key] = self.initial_tags[key]/self.initial_tags["total"]


    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        final_pos = []
        for word in sentence:
            pos = {}
            for tag in self.tags:
                pos[tag] = 100000
            for key in self.tags:
                if str(word) in self.word_dict[key]:
                    P = - math.log(self.word_dict[key][str(word)]/self.word_dict[key]["total"]) \
                        - math.log(self.priors[key])
                else:
                    P = -math.log(self.undef_prob)
                pos[key] = P
            final_pos.append(min(pos.items(), key=operator.itemgetter(1))[0])
        # print(final_pos)
        return final_pos

    def hmm_ve(self, sentence):
        ans = []
        first = True
        tau = 0.0
        prev_tag = None
        for word in sentence:
            max_tau = 1000
            max_prev_tag = None

            for key in self.tags:
                if first:
                    if word in self.word_dict[key]:
                        prob = -math.log(self.initial_state_distribution[key] *
                                         self.word_dict[key][word] / self.word_dict[key]["total"])
                    else:
                        prob = -math.log(self.initial_state_distribution[key]) - math.log(self.undef_prob)

                    if prob < max_tau:
                        max_tau = prob
                        max_prev_tag = key

                else:

                    if word in self.word_dict[key]:
                        if prev_tag in self.state_transitions[key]:
                            prob = - math.log(self.state_transitions[key][prev_tag] / 
                                              self.state_transitions[key]["total"]) \
                                   - math.log(self.word_dict[key][word] / 
                                              self.word_dict[key]["total"]) + tau
                        else:
                            prob = - math.log(self.undef_prob) \
                                   - math.log(self.word_dict[key][word] / 
                                              self.word_dict[key]["total"]) + tau
                    else:
                        if prev_tag in self.state_transitions[key]:
                            prob = - math.log(self.state_transitions[key][prev_tag] / 
                                              self.state_transitions[key]["total"]) \
                                   - math.log(self.undef_prob) + tau
                        else:
                            prob = - math.log(self.undef_prob) \
                                   - math.log(self.undef_prob) + tau

                    if prob < max_tau:
                        max_tau = prob
                        max_prev_tag = key

            tau = max_tau
            prev_tag = max_prev_tag
            ans.append(prev_tag)

            first = False
        return ans

    def hmm_viterbi(self, sentence):
        for key in self.tags:
            if sentence[0] in self.word_dict[key]:
                self.adjacency_list['0'][key] = (-math.log(self.initial_state_distribution[key]) - math.log(
                    self.word_dict[key][str(sentence[0])] / self.tag_dict[key]), 'Source')
            else:
                self.adjacency_list['0'][key] = (
                    -math.log(self.initial_state_distribution[key]) - math.log(self.undef_prob), 'Source')

        for i in range(1, len(sentence)):
            self.adjacency_list[str(i)] = {}

            for key in self.tags:
                adj_list = []
                for k in self.tags:
                    if sentence[i] in self.word_dict[key]:
                        if k not in self.state_transitions[key]:
                            self.state_transitions[key][k] = self.undef_prob
                        adj_list.append((self.adjacency_list[str(i - 1)][k][0] -
                                         math.log(self.state_transitions[key][k] /
                                                  self.state_transitions[key]['total']) -
                                         math.log(self.word_dict[key][str(sentence[i])] /
                                                  self.tag_dict[key]), k, sentence[i]))
                    else:
                        if k not in self.state_transitions[key]:
                            self.state_transitions[key][k] = self.undef_prob
                        adj_list.append((self.adjacency_list[str(i - 1)][k][0] -
                                         math.log(self.state_transitions[key][k] /
                                                  self.state_transitions[key]['total']) -
                                         math.log(self.undef_prob), k, sentence[i]))
                heapq.heapify(adj_list)
                self.adjacency_list[str(i)][key] = heapq.heappop(adj_list)
        PoS = []
        final_PoS = []
        for i in range(0, len(sentence), 1):
            if i == 0:
                PoS.append(min(self.adjacency_list[str(len(sentence) - 1 - i)].items(), key=operator.itemgetter(1)))
            else:
                PoS.append((PoS[-1][1][1], self.adjacency_list[str(len(sentence) - 1 - i)][PoS[-1][1][1]]))
        for values in PoS:
            final_PoS.append(values[0])
        # print(final_PoS)
        return final_PoS[::-1]

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
