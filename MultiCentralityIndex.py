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
          'StructuralHoles':nx.constraint, 'Pagerank':nx.pagerank_numpy, 
          'Betweenness':nx.betweenness_centrality,'Closeness':nx.closeness_centrality}

def cntV2DF(cntV):
    return pd.DataFrame(list(cntV.values()))
    
def checkNaNs(cntV,cnt):
    x = 1.0 if cnt == 'StructuralHoles' else 0.0
    cntV2 = { d:x  if np.isnan(cntV[d]) else cntV[d] for d in cntV}
    return cntV2
        
def calcCentrality(G,cnt):
    cntV = dict()
    node_dict = dict(list(dict(G.nodes(data=True)).values())[0])
    
    assert cnt in cntMap.keys(), ('calcCettrality: wrong centrality value or not implemented yet. Available:', list(cntMap.keys()))
    
    if node_dict.get(cnt) is None:
        cntV = cntMap[cnt](G)
        cntV = checkNaNs(cntV,cnt)        
        nx.set_node_attributes(G, cntV, cnt)
        
    return cntV2DF(dict(G.nodes(data=cnt)))

def getMatrixFeaturesGraph(G,cnts):
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
        val = calcCentrality(G,cnt)        
        #val = cntV2DF(cval)
        # Normalizing the data
        val = sc.fit_transform(val)
        mtxDoc[cnt] = val
    
    return mtxDoc


def getPC1(mtxFeatures, setCentralities):
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
    

def MCI_PC1(G, PC1, N):
    """
        Computes the Multi-Centrality index from:
        Vega-Oliveros, Didier; Gomes, Pedro; Milios, Evangelos; Berton, Lilian.
        "A multi-centrality index for graph-based keyword extraction"
        Information Processing & Management. 56. 102063. 10.1016/j.ipm.2019.102063.  
        The MCI is defined as the 1D principal component from a set of centrality measures.
        
        Parameters
        --------
                
            G : The networkx graph.
            PC1 : A datafram with the first principal component (columns are the component elements),
                and columns names are the respective centralities.
            N: The number of best MCI ranked node ids to return (-1 to return all nodes ranked) 
                  
                
        Returns:
        --------
        
            dataframe : the N best ranked nodes and scores according to the MCI
                
    """           
    
    sc = MinMaxScaler()
    #sc = StandardScaler()
    
    G_Words = pd.DataFrame()
    G_Words['Word'] = list(G.nodes)
    G_Words['MCI'] = np.zeros((len(G.nodes),1))
    for cnt in list(PC1):
        val = calcCentrality(G,cnt)
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
        
    #Number of requested nodes
    N = -1
    
    PC1 = getPC1(mtxFeatures, setCentralities)
    keywords = MCI_PC1(G, PC1, N)
    print(keywords)
    



