# -*- coding: utf-8 -*-
"""
Author: Didier A. Vega-Oliveros

"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler


cntMap = {'Degree':nx.degree_centrality, 'Eigenvector':nx.eigenvector_centrality_numpy, 
          'Pagerank':nx.pagerank_numpy, 'Closeness':nx.closeness_centrality,
          'StructuralHoles':nx.constraint,'Betweenness':nx.betweenness_centrality}


def cntV2DF(cntV):
    return pd.DataFrame(list(cntV.values()))
    
def checkNaNs(cntV,cnt):
    x = 1.0 if cnt == 'StructuralHoles' else 0.0
    cntV2 = { d:x  if np.isnan(cntV[d]) else cntV[d] for d in cntV}
    return cntV2
 
class MCI:
    
    def __init__(self):
        pass       
       
    def calcCentrality(self,G,cnt):
        """
        Computes the given centrality cnt in the graph G.
        
        Parameters
        --------
            G : The networkx graph.                
            cnts: The list of centrality measures to be considered.
                  
                
        Returns:
        --------
        
            dataframe: A dataframe in which the column corresponds to values of 
            the centrality measure
            
                
    """ 
    
        cntV = dict()
        node_dict = dict(list(dict(G.nodes(data=True)).values())[0])
        
        assert cnt in cntMap.keys(), ('calcCettrality: wrong centrality value or not implemented yet. Available:', list(cntMap.keys()))
        
        if node_dict.get(cnt) is None:
            cntV = cntMap[cnt](G)
            cntV = checkNaNs(cntV,cnt)        
            nx.set_node_attributes(G, cntV, cnt)
            
        return cntV2DF(dict(G.nodes(data=cnt)))


    def getMatrixFeaturesGraph(self,G,cnts):
        """
            Computes the matrix (dataframe) of features for a given single graph in the repository,
            This function is the equivalent to lines 5 - 11 of the Algorithm 1 from:
            "A multi-centrality index for graph-based keyword extraction"
            Information Processing & Management. 56. 102063. 10.1016/j.ipm.2019.102063.  
            
            
            Parameters
            --------
                G : The networkx graph.                
                cnts: The list of centrality measures to be considered.
                      
                    
            Returns:
            --------
            
                dataframe: A dataframe in whichs rows are nodes and columns the 
                    corresponding values of each centrality measure, which columns names 
                    the respective centralities.
                
                    
        """ 
        
        sc = MinMaxScaler()
        mtxDoc = pd.DataFrame()
        mtxDoc['Word'] = list(G.nodes)
        
        for cnt in cnts:
            val = self.calcCentrality(G,cnt) 
            # Normalizing the data
            val = sc.fit_transform(val)
            mtxDoc[cnt] = val
        
        return mtxDoc
    
    
    def getPC1(self, mtxFeatures, setCentralities=None):
        """
            Computes the first Principal Component from the table of centrality measures,
            As explained in Section '4.4. The MCI approach' from:
            "A multi-centrality index for graph-based keyword extraction"
            Information Processing & Management. 56. 102063. 10.1016/j.ipm.2019.102063.  
            
            
            Parameters
            --------
                    
                mtxFeatures : A dataframe in whichs rows are nodes and columns the 
                    corresponding values of each centrality measure.
                setCentralities: The list of centrality measures to be considered.
                      
                    
            Returns:
            --------
            
                dataframe : The first principal component, which columns names 
                    are the respective centralities and has only one row, the component values.
                
                    
        """           
        
        sc = StandardScaler()
        if setCentralities is None:
            setCentralities =  list(mtxFeatures.select_dtypes(include=['int',"float" ]).columns)
        A = mtxFeatures.loc[:,setCentralities]
        # normalize data
        A = pd.DataFrame(data = sc.fit_transform(A),  columns = list(A))    
        # create the PCA instance
        pca = PCA(n_components=1)
        # fit on data
        pca.fit(A)
        # access values and vectors    
        PC1 = pd.DataFrame(data=pca.components_, columns = list(A))
        return PC1
        
    def getPC1FromGraph(self, G, setCentralities=None):
        """
            Returns the first Principal Component from the the give graph by
            considering the a set of centrality features.
            
            "A multi-centrality index for graph-based keyword extraction"
            Information Processing & Management. 56. 102063. 10.1016/j.ipm.2019.102063.  
            
            
            Parameters
            --------
                    
                G : The networkx graph.
                setCentralities: The list of centrality measures to be considered.
                If None, then it considers all the possible centralities.
                      
                    
            Returns:
            --------
            
                dataframe : The first principal component, which columns names 
                    are the respective centralities and has only one row, the component values.
                
                    
        """    
        
        if setCentralities is None:
            setCentralities = list(cntMap.keys())
            
        mtxDoc = self.getMatrixFeaturesGraph(G,setCentralities)
        PC1 = self.getPC1(mtxDoc, setCentralities)
        return PC1
        
    def getMCI_PCA(self,G, PC1=None, setCetralities=None, N=-1):
        """
            Computes the Multi-Centrality index from:
            Vega-Oliveros, Didier; Gomes, Pedro; Milios, Evangelos; Berton, Lilian.
            "A multi-centrality index for graph-based keyword extraction"
            Information Processing & Management. 56. 102063. 10.1016/j.ipm.2019.102063.  
            The MCI is defined as the 1D principal component from a set of centrality measures.
            
            Parameters
            --------
                    
                G : The networkx graph.
                PC1 : A datafram with the first principal component 
                    (columns are the component elements and columns names are 
                    the respective centralities) extracted from a matrix of 
                    features of graphs.
                    
                    If PC1 = None, then the PC1 is calculated from the given graph G,
                    considering the set of Centralities
                
                setCentralities: (Optional) the list of centrality measures to be considered
                    in the case that none PC1 value is passed.
                
                N: The number of best MCI ranked node ids to return 
                    (-1 to return all nodes ranked) 
                      
                    
            Returns:
            --------
            
                dataframe : the N best ranked nodes and scores according to the MCI
                    
        """           
        
        assert PC1 is not None or setCetralities is not None, ('getMCI_PCA: Please provide either a PC1 or setCentralities parameter. Available centralities are:', list(cntMap.keys()))
        sc = MinMaxScaler()
        #sc = StandardScaler()
        
        G_Words = pd.DataFrame()
        G_Words['Word'] = list(G.nodes)
        G_Words['MCI'] = np.zeros((len(G.nodes),1))
        
        if PC1 is None:
            PC1 = self.getPC1FromGraph(G,setCetralities)
                
        for cnt in list(PC1):
            val = self.calcCentrality(G,cnt)
            
            # Normalizing the data
            val = sc.fit_transform(val)
            G_Words[cnt] = val
            G_Words['MCI'] += G_Words[cnt]*PC1.loc[0,cnt] 
        
        keynodes = G_Words.sort_values(by='MCI', ascending=False).loc[:,['Word','MCI']]
        
        if N == -1:
            return keynodes.reset_index(drop=True)
        else: 
            return keynodes.reset_index(drop=True).head(N)
       
  
 
if __name__ == "__main__":       

    #EXEMPLE OF USE
    #setCentralities = ['Degree','Eigenvector','StructuralHoles']
    setCentralities = ['Degree','Pagerank','Eigenvector','StructuralHoles']
    
    #PRELOADED MATRIX OF FEATURES (CENTRALITIES) FROM A RESPOSITORY
    folderData = os.getcwd()
    df0 = pd.read_csv(os.path.join(folderData, "tableCentralityMeasuresClasses.txt"), sep = '\t')
    mtxFeatures = df0.drop(['FileName', 'Word', 'WordLength','Class'], axis=1) 
    
    #Loading a graph-of-word constructed from a file of the respository 
    G=nx.read_edgelist("edgelist_art_and_culture-20914080.txt")
    nx.draw_networkx(G)
        
    #Number of requested nodes -1, wich means all the nodes
    N = -1
    
    mc = MCI()
    
    PC1 = mc.getPC1(mtxFeatures, setCentralities)
    keywords = mc.getMCI_PCA(G, PC1, N)[:15]
    print(keywords)
    print(" ... ")
    #display(keywords)
    
    



