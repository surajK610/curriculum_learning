"""
author: @john-hewitt
https://github.com/john-hewitt/structural-probes

Contains classes describing linguistic tasks of interest on annotated data.
"""

import numpy as np
import torch


class Task:
    """Abstract class representing a linguistic task mapping texts to labels."""
    
    @staticmethod
    def labels(observation):
        """Maps an observation to a matrix of labels.
    
    Should be overriden in implementing classes.
    """
        raise NotImplementedError


class ParseDistanceTask(Task):
    """Maps observations to dependency parse distances between words."""

    @staticmethod
    def labels(observation):
        """Computes the distances between all pairs of words; returns them as a torch tensor.

    Args:
        observation: a single Observation class for a sentence:
    Returns:
        A torch tensor of shape (sentence_length, sentence_length) of distances
        in the parse tree as specified by the observation annotation.
    """
        sentence_length = len(
            observation[0]
        )  # All observation fields must be of same length
        distances = torch.zeros((sentence_length, sentence_length))
        for i in range(sentence_length):
            for j in range(i, sentence_length):
                i_j_distance = ParseDistanceTask.distance_between_pairs(
                    observation, i, j
                )
                distances[i][j] = i_j_distance
                distances[j][i] = i_j_distance
        return distances

    @staticmethod
    def distance_between_pairs(observation, i, j, head_indices=None):
        """Computes path distance between a pair of words

    TODO: It would be (much) more efficient to compute all pairs' distances at once;
        this pair-by-pair method is an artefact of an older design, but
        was unit-tested for correctness... 

    Args:
        observation: an Observation namedtuple, with a head_indices field.
            or None, if head_indies != None
        i: one of the two words to compute the distance between.
        j: one of the two words to compute the distance between.
        head_indices: the head indices (according to a dependency parse) of all
            words, or None, if observation != None.

    Returns:
        The integer distance d_path(i,j)
    """
        if i == j:
            return 0
        if observation:
            head_indices = []
            number_of_underscores = 0
            for elt in observation.head_indices:
                if elt == "_":
                    head_indices.append(0)
                    number_of_underscores += 1
                else:
                    head_indices.append(int(elt) + number_of_underscores)
        i_path = [i + 1]
        j_path = [j + 1]
        i_head = i + 1
        j_head = j + 1
        while True:
            if not (i_head == 0 and (i_path == [i + 1] or i_path[-1] == 0)):
                i_head = head_indices[i_head - 1]
                i_path.append(i_head)
            if not (j_head == 0 and (j_path == [j + 1] or j_path[-1] == 0)):
                j_head = head_indices[j_head - 1]
                j_path.append(j_head)
            if i_head in j_path:
                j_path_length = j_path.index(i_head)
                i_path_length = len(i_path) - 1
                break
            elif j_head in i_path:
                i_path_length = i_path.index(j_head)
                j_path_length = len(j_path) - 1
                break
            elif i_head == j_head:
                i_path_length = len(i_path) - 1
                j_path_length = len(j_path) - 1
                break
        total_length = j_path_length + i_path_length
        return total_length


class ParseDepthTask:
    """Maps observations to a depth in the parse tree for each word"""

    @staticmethod
    def labels(observation):
        """Computes the depth of each word; returns them as a torch tensor.

    Args:
        observation: a single Observation class for a sentence:
    Returns:
        A torch tensor of shape (sentence_length,) of depths
        in the parse tree as specified by the observation annotation.
    """
        sentence_length = len(
            observation[0]
        )  # All observation fields must be of same length
        depths = torch.zeros(sentence_length)
        for i in range(sentence_length):
            depths[i] = ParseDepthTask.get_ordering_index(observation, i)
        return depths

    @staticmethod
    def get_ordering_index(observation, i, head_indices=None):
        """Computes tree depth for a single word in a sentence

    Args:
        observation: an Observation namedtuple, with a head_indices field.
            or None, if head_indies != None
        i: the word in the sentence to compute the depth of
        head_indices: the head indices (according to a dependency parse) of all
            words, or None, if observation != None.

    Returns:
        The integer depth in the tree of word i
    """
        if observation:
            head_indices = []
            number_of_underscores = 0
            for elt in observation.head_indices:
                if elt == "_":
                    head_indices.append(0)
                    number_of_underscores += 1
                else:
                    head_indices.append(int(elt) + number_of_underscores)
        length = 0
        i_head = i + 1
        while True:
            i_head = head_indices[i_head - 1]
            if i_head != 0:
                length += 1
            else:
                return length
              
              
class CPosTask(Task):
  
  @staticmethod
  def labels(observation):
    """Maps an observation to a matrix of labels.
    Should be overriden in implementing classes.
    """
    ud_cpos_categories = {
    "ADJ": 0,    # Adjective
    "ADP": 1,    # Adposition
    "ADV": 2,    # Adverb
    "AUX": 3,    # Auxiliary verb
    "CCONJ": 4,   # Coordinating conjunction
    "DET": 5,    # Determiner
    "INTJ": 6,   # Interjection
    "NOUN": 7,   # Noun
    "NUM": 8,    # Numeral
    "PART": 9,   # Particle
    "PRON": 10,  # Pronoun
    "PROPN": 11, # Proper noun
    "PUNCT": 12, # Punctuation
    "SCONJ": 13, # Subordinating conjunction
    "SYM": 14,   # Symbol
    "VERB": 15,  # Verb
    "X": 16,      # Other
    "_": 16
  }
    # print(observation.upos_sentence)
    return torch.tensor([ud_cpos_categories[elt] for elt in observation.upos_sentence])

class FPosTask(Task):
  
  @staticmethod
  def labels(observation):
    """Maps an observation to a matrix of labels.
    Should be overriden in implementing classes.
    """
    penn_treebank_pos_tags = {
      "CC": 0,   # Coordinating conjunction
      "CD": 1,   # Cardinal number
      "DT": 2,   # Determiner
      "EX": 3,   # Existential there
      "FW": 4,   # Foreign word
      "HYPH": 5, # Hyphen
      "IN": 6,   # Preposition or subordinating conjunction
      "JJ": 7,   # Adjective
      "JJR": 8,  # Adjective comparative
      "JJS": 9,  # Adjective superlative
      "LS": 10,   # List item marker
      "MD": 11,  # Modal
      "NN": 12,  # Noun singular or mass
      "NNS": 13, # Noun plural
      "NNP": 14, # Proper noun singular
      "NNPS": 15, # Proper noun plural
      "PDT": 16, # Predeterminer
      "POS": 17, # Possessive ending
      "PRP": 18, # Personal pronoun
      "PRP$": 19, # Possessive pronoun
      "RB": 20,  # Adverb
      "RBR": 21, # Adverb comparative
      "RBS": 22, # Adverb superlative
      "RP": 23,  # Particle
      "SYM": 24, # Symbol
      "TO": 25,  # to
      "UH": 26,  # Interjection
      "VB": 27,  # Verb base form
      "VBD": 28, # Verb past tense
      "VBG": 29, # Verb gerund or present participle
      "VBN": 30, # Verb past participle
      "VBP": 31, # Verb non-3rd person singular present
      "VBZ": 32, # Verb 3rd person singular present
      "WDT": 33, # Wh-determiner
      "WP": 34,  # Wh-pronoun
      "WP$": 35, # Possessive wh-pronoun
      "WRB": 36  # Wh-adverb
    }
    
    return torch.tensor([penn_treebank_pos_tags[elt] if elt in penn_treebank_pos_tags else 37 for elt in observation.xpos_sentence])

class DepTask(Task):
  
  @staticmethod
  def labels(observation):
    """Maps an observation to a matrix of labels.
    Should be overriden in implementing classes.
    """
    synct_dep_tags = {'acl': 0,
        'acl:relcl': 1,
        'advcl': 2,
        'advcl:relcl': 3,
        'advmod': 4,
        'amod': 5,
        'appos': 6,
        'aux': 7,
        'aux:pass': 8,
        'case': 9,
        'cc': 10,
        'cc:preconj': 11,
        'ccomp': 12,
        'compound': 13,
        'compound:prt': 14,
        'conj': 15,
        'cop': 16,
        'csubj': 17,
        'csubj:outer': 18,
        'csubj:pass': 19,
        'dep': 20,
        'det': 21,
        'det:predet': 22,
        'discourse': 23,
        'dislocated': 24,
        'expl': 25,
        'fixed': 26,
        'flat': 27,
        'goeswith': 28,
        'iobj': 29,
        'list': 30,
        'mark': 31,
        'nmod': 32,
        'nmod:desc': 33,
        'nmod:npmod': 34,
        'nmod:poss': 35,
        'nmod:tmod': 36,
        'nsubj': 37,
        'nsubj:outer': 38,
        'nsubj:pass': 39,
        'nummod': 40,
        'obj': 41,
        'obl': 42,
        'obl:agent': 43,
        'obl:npmod': 44,
        'obl:tmod': 45,
        'orphan': 46,
        'parataxis': 47,
        'punct': 48,
        'reparandum': 49,
        'root': 50,
        'vocative': 51,
        'xcomp': 52,
        '_': 53}
    # filter if no label
    return torch.tensor([synct_dep_tags[elt] for elt in observation.governance_relations])

class NerTask(Task):
  
  @staticmethod
  def labels(observation):
    """Maps an observation to a matrix of labels.
    Should be overriden in implementing classes.
    """
    ner_categories = {
    'CARDINAL': 0,
    'DATE': 1,
    'EVENT': 2,
    'FAC': 3,
    'GPE': 4,
    'LANGUAGE': 5,
    'LAW': 6,
    'LOC': 7,
    'MONEY': 8,
    'NORP': 9,
    'ORDINAL': 10,
    'ORG': 11,
    'PERCENT': 12,
    'PERSON': 13,
    'PRODUCT': 14,
    'QUANTITY': 15,
    'TIME': 16,
    'WORK_OF_ART': 17,
    '*': 18
    }
    # print(observation.upos_sentence)
    return torch.tensor([ner_categories[elt] for elt in observation.ner])

class PhrStartTask(Task): 
  @staticmethod
  def labels(observation):
    """Maps an observation to a matrix of labels.
    Should be overriden in implementing classes.
    """
    start_map = {
    "NS": 0, 
    "S": 1,   
    }
    return torch.tensor([start_map[elt] for elt in observation.phrase_start])

class PhrEndTask(Task): 
  @staticmethod
  def labels(observation):
    """Maps an observation to a matrix of labels.
    Should be overriden in implementing classes.
    """
    end_map = {
    "NE": 0, 
    "E": 1,   
    }
    return torch.tensor([end_map[elt] for elt in observation.phrase_end])