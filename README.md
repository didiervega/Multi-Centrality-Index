# Multi-centrality index

## How to use it and examples 

This is a notebook showing how to use the code of the proposed method for multicentrality index, which was employed for the analysis in [1]. The code is in Python3, and some toolboxes are necessary to run the commands. 

The following very common packages are necessary for running the code:

* numpy
* pandas
* networkx
* sklearn

``` python
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
```

Then, you can run the example for a preloaded matrix of features (centralities) from a previous constructed graph-of-words (network) with the co-occurrence approach. 

The result is each word of the graph sorted by the corresponding Multi-centrality Index (**MCI**) value. 
In this example, the MCI is the combination of these centrality measures: [*'Degree','Pagerank','Eigenvector','StructuralHoles'*]

Top words are considered the keywords of the text.

You run the code as follow:

```python
python MultiCentralityIndex.py
```


                Word       MCI
    0          MAMET  1.501088
    1           PLAY  1.484412
    2       DIRECTOR  0.968968
    3      ANARCHIST  0.887786
    4        THEATER  0.712872
    5       PULITZER  0.647991
    6         LONDON  0.635058
    7          GOOLD  0.619282
    8           YEAR  0.605723
    9      GLENGARRY  0.572361
    10         DEBUT  0.530749
    11         DAVID  0.311259
    12          YORK  0.225655
    13   PRIZEWINNER  0.215347
    14         PRIZE  0.208866
    15        RUPERT  


<b>              ...              </b>


![png](output_8_0.png)


Besides, you can import and use the code as your necessity. For example, lets calculate the MCI for the <a href='http://www-personal.umich.edu/~mejn/centrality/' target='_blank'>Coauthorships in network science</a> <table><tr><td><span><b> A figure depicting the largest component of this network</b></span> Extracted from Prof. Newman <a href='http://www-personal.umich.edu/~mejn/netdata' target='_blank'> Web site</a></td><td><img src='http://www-personal.umich.edu/~mejn/centrality/labeleds.png'></td></tr></table>


``` python
import MultiCentralityIndex as mc
import networkx as nx

#loading the netscience graph
G = nx.read_gml('netscience.gml',label='label')
node_size=[float(G.degree(v)) for v in G]

#Showing the graph of the full network
nx.draw_networkx(G, arrows=True, node_size=20, node_color=node_size,edge_color='grey',alpha=.5,with_labels=False)
```

![png](output_11_0.png)


Now, let's define the set of centrality measures to be calculated as
``` python 
setCentralities = ['Degree','Pagerank','Eigenvector','StructuralHoles','Closeness', 'Betweenness']
```
In this example we'll calculate the MCI for a single graph (network). First, we call the ```getMatrixFeaturesGraph``` function getting the matrix of centrality measures of the graph (``mtxDoc``). Then, we call the ```getPC1``` function for computing the first Principal Component (PC1) of the graph.

**Note**: In the case of ref[1], we calculated the matrix of features from the set of graphs in a repository and, then, we computed the first Principal Component (```getPC1``` function) from this matrix of features of the entire repository.



``` python
mtxDoc = mc.getMatrixFeaturesGraph(G,setCentralities)
print(mtxDoc)
```


                     Word    Degree  Pagerank   Eigenvector  StructuralHoles  \
    0         ABRAMSON, G  0.058824  0.139844  4.162192e-16         0.887188   
    1         KUPERMAN, M  0.088235  0.220900  4.325825e-16         0.508935   
    2          ACEBRON, J  0.117647  0.142576  4.613181e-16         0.656586   
    3          BONILLA, L  0.117647  0.142576  4.617155e-16         0.656586   
    4     PEREZVICENTE, C  0.117647  0.142576  4.623615e-16         0.6565


<b>              ...              </b>


```python
PC1 = mc.getPC1(mtxDoc, setCentralities)
print(PC1)
```

         Degree  Pagerank  Eigenvector  StructuralHoles  Closeness  Betweenness
    0  0.535797  0.434088     0.202644        -0.494563   0.329897     0.360556
    

Now, we can calculate the MCI for the graph calling the function 

``` python 
N = 10
MCI_PC1(G, PC1, N)
print(MCI)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Word</th>
      <th>MCI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NEWMAN, M</td>
      <td>1.550027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BARABASI, A</td>
      <td>1.330304</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JEONG, H</td>
      <td>1.221634</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PASTORSATORRAS, R</td>
      <td>1.043823</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SOLE, R</td>
      <td>1.040874</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BOCCALETTI, S</td>
      <td>0.978564</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MORENO, Y</td>
      <td>0.903129</td>
    </tr>
    <tr>
      <th>7</th>
      <td>HOLME, P</td>
      <td>0.871221</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CALDARELLI, G</td>
      <td>0.808137</td>
    </tr>
    <tr>
      <th>9</th>
      <td>VESPIGNANI, A</td>
      <td>0.807601</td>
    </tr>
  </tbody>
</table>
</div>


# References
You can use this code as it is for academic purpose. If you found it useful for your research, we appreciate your reference to our work _A multi-centrality index for graph-based keyword extraction_:

[1] Didier A. Vega-Oliveros, Pedro Spoljaric Gomes, Evangelos E. Milios, Lilian Berton. Information Processing & Management, V. 56, I. 6, November 2019, 102063. https://doi.org/10.1016/j.ipm.2019.102063



```python

```
