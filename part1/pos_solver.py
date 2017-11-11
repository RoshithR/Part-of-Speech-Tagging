###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
####

import random
import math
import operator


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    def __init__(self):
        #tag_dict maintains count of occurences of each tag and the total number of tags
        self.tag_dict = {"adj": 0, "adv": 0, "adp": 0, "conj": 0, "det": 0, "noun": 0, "num": 0, "pron": 0, "prt": 0,"verb": 0, "x": 0, ".": 0, "total": 0}
        #word_dict maintains occurence of each word within a specific tag
        self.word_dict = {"adj": {'total':0}, "adv": {'total':0}, "adp": {'total':0}, "conj": {'total':0}, "det": {'total':0}, "noun": {'total':0}, "num": {'total':0}, "pron": {'total':0},"prt": {'total':0}, "verb": {'total':0}, "x": {'total':0}, ".": {'total':0}}
        #priors computes the number of time a tag occurs in testing data i.e. #adj/total
        self.priors={}
        #intial_tags stores the number of times a part of speech tag occurs at the start of a sentence
        self.initial_tags={"adj": 0, "adv": 0, "adp": 0, "conj": 0, "det": 0, "noun": 0, "num": 0, "pron": 0, "prt": 0,"verb": 0, "x": 0, ".": 0, "total": 0}
        #initial probabilities are computed and stored here as initial_tag=def/total number of initial tags from above dictionary
        self.initial_state_distribution={"adj": 0, "adv": 0, "adp": 0, "conj": 0, "det": 0, "noun": 0, "num": 0, "pron": 0, "prt": 0,"verb": 0, "x": 0, ".": 0, "total": 0}
        #number of times a given tag(the key here) is preceeded by another PoS tag. To compute P(X2=Det|X1=Noun)
        self.state_transitions = {"adj": {'total':0}, "adv": {'total':0}, "adp": {'total':0}, "conj": {'total':0}, "det": {'total':0}, "noun": {'total':0}, "num": {'total':0}, "pron": {'total':0},"prt": {'total':0}, "verb": {'total':0}, "x": {'total':0}, ".": {'total':0}}
        #Graph for viterbi algorithm
        self.adjacency_list={'Source':[]}
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labelingif str(word) in self.word_dict["adj"]:
    def posterior(self, sentence, label):
        return 0

    # Do the training!
    #
    def train(self, data):

        for line in data:
            self.initial_tags[line[1][0]]+=1
            self.initial_tags['total']+=1
            for word,tag in zip(line[0],line[1]):
                self.tag_dict[tag]+=1
                self.tag_dict["total"]+=1

                if word in self.word_dict[tag]:
                    self.word_dict[tag][word]+=1
                    self.word_dict[tag]['total']+=1
                else:
                    self.word_dict[tag][word]=1
                    self.word_dict[tag]['total'] += 1
            #training transition probabilities
            for index,tag in enumerate(line[1]):
                if index>0:
                    if line[1][index-1] in self.state_transitions[tag]:
                        self.state_transitions[tag][line[1][index-1]]+=1
                        self.state_transitions[tag]['total'] += 1
                    else:
                        self.state_transitions[tag][line[1][index - 1]] = 1
                        self.state_transitions[tag]['total'] += 1


        for key in ["adj", "adv", "adp", "conj", "det", "noun", "num", "pron", "prt", "verb", "x", "."]:
            self.priors[key]=self.tag_dict[key]/self.tag_dict["total"]
            self.initial_state_distribution[key]=self.initial_tags[key]/self.initial_tags['total']

        print(self.tag_dict)
        print(self.word_dict)
        print(self.priors)
        print(self.initial_tags)
        print(self.initial_state_distribution)
        print(self.state_transitions)

        # word_dict={"adj":{"total":0}, "adv":{"total":0}, "adp":{"total":0}, "conj":{"total":0}, "det":{"total":0}, "noun":{"total":0}, "num":{"total":0}, "pron":{"total":0}, "prt":{"total":0}, "verb":{"total":0}, "x":{"total":0}, ".":{"total":0}}


    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        final_pos=[]
        for word in sentence:
            pos={"adj":0, "adv":0, "adp":0, "conj":0, "det":0, "noun":0, "num":0, "pron":0, "prt":0, "verb":0, "x":0, ".":0}
            for key in ["adj","adv","adp","conj","det","noun","num","pron","prt","verb","x","."]:
                if str(word) in self.word_dict[key]:
                    P=(self.word_dict[key][str(word)]/self.word_dict[key]['total'])*self.priors[key]
                else:
                    P=0
                pos[key]=P
            final_pos.append(max(pos.items(), key=operator.itemgetter(1))[0])
        print(final_pos)
        return (final_pos)

    def hmm_ve(self, sentence):

        return [ "noun" ] * len(sentence)

    def hmm_viterbi(self, sentence):
        # for word in sentence:
        #     for key in ["adj","adv","adp","conj","det","noun","num","pron","prt","verb","x","."]:
        #         self.adjacency_list['Source']=(key,-math.log(self.initial_state_distribution[key])-math.log(self.word_dict[key][str(word)]))
        return [ "noun" ] * len(sentence)


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

