# -*- coding: utf-8 -*-
"""
Author: Didier A. Vega-Oliveros

"""

import os
import re
import numpy as np
import pandas as pd
import networkx as nx
from nltk import sent_tokenize
from nltk.tokenize import wordpunct_tokenize
import string
import en_core_web_sm
from itertools import chain, groupby
from collections import defaultdict
import MultiCentralityIndex as MCI

#import pt_core_news_lg
#nlp = pt_core_news_lg.load()
 
class TextMiner:
    
    def __init__(self,
                 max_length_sent=100000,
                 min_length_sent=5,
                 punctuations=None,
                 nlp = None,                 
                 ):
        
        self.candi_pos = ['NOUN', 'PROPN', 'VERB','ADJ'] 
        self.stop_pos = ['NUM', 'ADV']
        self.span = 3
        self.mc = MCI.MCI()
        
        self.max_length_sent=max_length_sent
        self.min_length_sent=min_length_sent
        
        # If punctuations are not provided, ignore all punctuation symbols.
        self.punctuations = punctuations
        if self.punctuations is None:
            self.punctuations = string.punctuation
        
        self.index = None
        self.rank_list = None
        self.ranked_phrases = None
        
        self.nlp = nlp
        if self.nlp is None:
            self.nlp = en_core_web_sm.load()

    
    def clean_spaces(self, s):
        s = s.replace('\r', '')
        s = s.replace('\t', ' ')
        s = s.replace('\n', ' ')
        return s

    def remove_noisy(self, content):
        """Remove brackets symbols"""
        p1 = re.compile(r'（[^）]*）')
        p2 = re.compile(r'\([^\)]*\)')
        p3 = re.compile("(\\d|\\W)+")
        p4 = re.compile("&lt;/?.*?&gt;")
        p5 = re.compile(r'https?:\S+')
        p6 = re.compile(r'www\S+')    
        return p3.sub(" ",p2.sub('', p1.sub('', p4.sub(" &lt;&gt; ", p6.sub('',p5.sub('',content))))))
    
    def remove_noisy2(self, content):
        """Remove brackets"""
        p1 = re.compile(r'（[^）]*）')
        p2 = re.compile(r'\([^\)]*\)')
        p4 = re.compile("&lt;/?.*?&gt;")
        p5 = re.compile(r'https?:\S+')
        p6 = re.compile(r'www\S+') 
        p7 = re.compile(r'-') 
        return p2.sub(', ', p1.sub(', ', p4.sub(", ", p6.sub(', ',p5.sub(', ',p7.sub(',',content))))))


    
    def _generate_phrases(self, sentences):
        """Method to generate contender phrases given the sentences of the text
        document.
        :param sentences: List of strings where each string represents a
                          sentence which forms the text.
        :return: Set of string tuples where each tuple is a collection
                 of words forming a contender phrase.
        """
        ### Extracted from rake-nltk project: https://github.com/csurfer/rake-nltk
        
        phrase_list = set()
        # Create contender phrases from sentences.
        for sentence in sentences:
            word_list = [word.lower() for word in wordpunct_tokenize(sentence)]
            phrase_list.update(self._get_phrase_list_from_words(word_list))
        return phrase_list
    
    
    def _get_phrase_list_from_words(self, word_list):
        """Method to create contender phrases from the list of words that form
        a sentence by dropping stopwords and punctuations and grouping the left
        words into phrases. Only phrases in the given length range (both limits
        inclusive) would be considered to build co-occurrence matrix. Ex:
        Sentence: Red apples, are good in flavour.
        List of words: ['red', 'apples', ",", 'are', 'good', 'in', 'flavour']
        List after dropping punctuations.
        List of words: ['red', 'apples', *, 'are', 'good', 'in', 'flavour']
        List of phrases: [('red', 'apples'), ('are', 'good', 'in', 'flavour')]
        List of phrases with a correct length:
        For the range [1, 2]: [('red', 'apples'),]
        For the range [1, 4]: [('red', 'apples'), ('are', 'good', 'in', 'flavour'),]
        For the range [4, 4]: [('are', 'good', 'in', 'flavour'),]
        :param word_list: List of words which form a sentence when joined in
                          the same order.
        :return: List of contender phrases that are formed after dropping
                 stopwords and punctuations.
        """
        #HACKED groups:
        ### Adapted from rake-nltk project: https://github.com/csurfer/rake-nltk
        groups = groupby(word_list, lambda x: x not in self.punctuations)
        phrases = [tuple(group[1]) for group in groups if group[0]]
        return list(
            filter(
                lambda x: self.min_length_sent <= len(x) <= self.max_length_sent, phrases)
            )

    
    def _keywords_mtxDoc_graph(self, word_list, setCentralities=None):
                
        if setCentralities is None:
            setCentralities = list(MCI.cntMap.keys())[:4]
            
        g = nx.Graph()
        cm = defaultdict(int)
        for i, word in enumerate(word_list): # word_list = [['previous', 'ADJ'], ['rumor', 'NOUN']]
            if word[1] in self.candi_pos and len(word[0]) > 1: # word = ['previous', 'ADJ']
                for j in range(i + 1, i + self.span):
                    if j >= len(word_list):
                        break
                    if word_list[j][1] not in self.candi_pos or word_list[j][1] in self.stop_pos or len(word_list[j][0]) < 2:
                        continue
                    pair = tuple((word[0], word_list[j][0]))
                    cm[(pair)] +=  1

        # cm = {('was', 'prison'): 1, ('become', 'prison'): 1}
        for terms, w in cm.items():
            g.add_node(terms[0])
            g.add_node(terms[1])
            g.add_edge(terms[0], terms[1], weight=w)
            #print(terms[0], terms[1],w)
        
        mtxDoc = self.mc.getMatrixFeaturesGraph(g,setCentralities)    
        return mtxDoc, g
    

    def _keywords_MCI(self, word_list, num_keywords, mtxDoc=None, setCentralities=None):        
        
        mxRes, g = self._keywords_mtxDoc_graph(word_list, setCentralities=setCentralities)
        if mtxDoc is None:
            mtxDoc = mxRes 
            
        PC1 = self.mc.getPC1(mtxDoc,setCentralities)
        nodes_rank = self.mc.getMCI_PCA(g, PC1, num_keywords)
        
        return nodes_rank

    
    def rank_phrases(self, phrase_list):
        """Method to rank each contender phrase using the formula
              phrase_score = sum of scores of words in the phrase.
              word_score = MCI(w) where MCI is the centrality index.
        :param phrase_list: List of List of strings where each sublist is a
                            collection of words which form a contender phrase.
        """
        ### Adapted from rake-nltk project: https://github.com/csurfer/rake-nltk
        
        assert self.index is not None , ('Please first call the keywords method and then you can get the key phrases.') 
        self.rank_list = []
        lkeywords = list(self.index.keys())
        
        for phrase in phrase_list:
            rank = 0.0
            for word in phrase:
                if word in lkeywords:
                    rank += 1.0 * self.index[word]
                
            self.rank_list.append((rank, " ".join(phrase)))

        self.rank_list.sort(reverse=True)
        self.ranked_phrases = [ph[1] for ph in self.rank_list]
    
    
    def get_ranked_phrases(self):
        """Method to fetch ranked phrases strings.
        :return: List of strings (phrases) where each string represents a 
            key phrases from the text where the keywords were computed.
        """
              
        return self.ranked_phrases
   
     
    def get_mtxDoc_from_collection(self, collection, setCentralities=None):
        """Method to calculate the matrix of features from a collection of texts.
         :param setCentralities: The list of centrality measures to be considered.
                If None, then it consider the first 4 centralities.
        :return: A dataframe in which rows are words and columns the 
                    corresponding values of each centrality measure. Columns names are 
                    the respective centralities.
        """
        if not collection:
            return []
        
        mtxDoc = pd.DataFrame()
        
        for content in collection:
            words_postags = []
        
            # 01 remove linebreaks and brackets
            content = self.remove_noisy(content)
            content = self.clean_spaces(content).lower()
        
            # 02 split to sentences
            doc = self.nlp(content)
            for i, sent in enumerate(doc.sents):
                words_postags.extend([[token.text, token.pos_] for token in sent])
        
            dfResults, _ = self._keywords_mtxDoc_graph(words_postags,setCentralities=setCentralities)
            mtxDoc = mtxDoc.append(dfResults,ignore_index=True)
            
        return mtxDoc
   
    
    def get_keywords_MCI_from_text(self, content, mtxDoc=None, setCentralities=None, numberKeyWords=10):
        """
            Find the Key words from the given text by considering the Multi-Centrality index. 
            The MCI is the dimension reduction from set of centrality measures.
            
            Parameters
            --------
                    
                content: The text to be analyzed.
                mtxDoc : From a collection of texts, it is a dataframe
                    in which rows are nodes (words) and columns the corresponding
                    values of each centrality measure. Columns names are
                    the respective centralities. (optional)
                numberKeyWords: The number of best MCI ranked words to return. 
                    If -1, it returns all nodes ranked. 
                :param setCentralities: The list of centrality measures to be considered.
                If None, then it consider the first 4 centralities.
                      
                    
            Returns:
            --------
            
                dataframe : the N best ranked nodes and scores according to the MCI
                    
        """           
        if not content:
            return []
                
        numberKeyWordsCut = -1            
        words_postags = []  # token and its POS tag
        
        # 01 remove linebreaks and brackets
        content = self.remove_noisy2(content)
        content = self.clean_spaces(content).lower()
        content = content.replace('\n', ' . ').replace('--:--',' ').replace('/',' ').replace(':',' , ').replace(',,',' , ')
        
        # 02 split to sentences
        doc = self.nlp(content)
        sentences = list()
        
        for i, sent in enumerate(doc.sents):
            sentences.append(sent.text)            
            words_postags.extend([[token.text, token.pos_] for token in sent])
                            
        # 03 get keywords
        keywordsMCI = self._keywords_MCI(words_postags,numberKeyWordsCut,
                                         mtxDoc=mtxDoc,setCentralities=setCentralities)     
        self.index = defaultdict(float)
        for row in keywordsMCI.itertuples():
            self.index[row.Word] = row.MCI
                
        # 04 get keyphrases
        phrase_list = self._generate_phrases(sentences)
        self.rank_phrases(phrase_list)    
        
        return keywordsMCI.head(numberKeyWords)
  
 
if __name__ == "__main__":       

    #Example of use
    Miner = TextMiner(punctuations=None,min_length_sent=7, nlp=en_core_web_sm.load())
    Miner.candi_pos = ['NOUN', 'PROPN', 'ADJ']  
    N = 10
    #Loading three stories of Edgar Allan Poe
    from data.content import content
        
    #1. Considering a single text    
    keywords = Miner.get_keywords_MCI_from_text(content[0],numberKeyWords=N)
    print(keywords)  
    print('\n \t Key sentences \n')
    for sentence in Miner.get_ranked_phrases()[:N]:
        print(sentence,'\n ----------')
    
    print('   ========   ')   
    
    #2. Considering a collection of texts
    mtxDoc = Miner.get_mtxDoc_from_collection(content,setCentralities=
                                              ['Degree', 
                                               'Pagerank', 
                                               'StructuralHoles'])
    
    keywords = Miner.get_keywords_MCI_from_text(content[0],mtxDoc=mtxDoc,
                                                numberKeyWords=N)
    print(keywords)  
    print('\n \t Key sentences \n')
    for sentence in Miner.get_ranked_phrases()[:N]:
        print(sentence,'\n ----------')
    print('-------')
    
    



