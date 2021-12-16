# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:44:12 2020

@author: USUARIO
"""

import operator
import random 
from random import sample
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt
from operator import itemgetter

class request:

##############################--URLLC---#############################################################################
    B = nx.Graph()
    B.add_node(1)
    B.nodes[1]['Type'] = 'EDGE'
    B.nodes[1]['vnf'] = 'AMF'
    B.nodes[1]['cpu'] = 5 
    B.add_node(2)
    B.nodes[2]['Type'] = 'EDGE'
    B.nodes[2]['vnf'] = 'SMF'
    B.nodes[2]['cpu'] = 5 
    B.add_node(3)
    B.nodes[3]['Type'] = 'EDGE'
    B.nodes[3]['vnf'] = 'UPF'
    B.nodes[3]['cpu'] = 5 
    B.add_node(4)
    B.nodes[4]['Type'] = 'CORE'
    B.nodes[4]['vnf'] = 'UPF-BK'
    B.nodes[4]['cpu'] = 5
    B.add_node(5)
    B.nodes[5]['Type'] = 'CORE'
    B.nodes[5]['vnf'] = 'UPF-BK'
    B.nodes[5]['cpu'] = 5 
    B.add_node(6)
    B.nodes[6]['Type'] = 'CORE'
    B.nodes[6]['vnf'] = 'AMF-BK'
    B.nodes[6]['cpu'] = 5 
    B.add_node(7)
    B.nodes[7]['Type'] = 'CORE'
    B.nodes[7]['vnf'] = 'SMF-BK'
    B.nodes[7]['cpu'] = 5 
    
    B.add_edge(1,2)
    B.edges[(1,2)]['bw'] = 2  
    B.add_edge(2,3)
    B.edges[(2,3)]['bw'] = 2
    B.add_edge(3,4)
    B.edges[(3,4)]['bw'] = 2
    B.add_edge(3,5)
    B.edges[(3,5)]['bw'] = 2
    B.add_edge(4,5)
    B.edges[(4,5)]['bw'] = 2
    B.add_edge(1,6)
    B.edges[(1,6)]['bw'] = 2
    B.add_edge(2,7)
    B.edges[(2,7)]['bw'] = 2
    #nx.draw(B,with_labels=True)
    
    #######################################---EMMB----###################################################################
    E = nx.Graph()
    E.add_node(1)
    E.nodes[1]['Type'] = 'CORE'
    E.nodes[1]['vnf'] = 'AMF'
    E.nodes[1]['cpu'] = 5 
    E.add_node(2)
    E.nodes[2]['Type'] = 'CORE'
    E.nodes[2]['vnf'] = 'AMF'
    E.nodes[2]['cpu'] = 5 
    E.add_node(3)
    E.nodes[3]['Type'] = 'CORE'
    E.nodes[3]['vnf'] = 'SMF'
    E.nodes[3]['cpu'] = 5 
    E.add_node(4)
    E.nodes[4]['Type'] = 'CORE'
    E.nodes[4]['vnf'] = 'UPF'
    E.nodes[4]['cpu'] = 5
    
    E.add_edge(1,3)
    E.edges[(1,3)]['bw'] = 2  
    E.add_edge(2,3)
    E.edges[(2,3)]['bw'] = 2
    E.add_edge(3,4)
    E.edges[(3,4)]['bw'] = 2
    #######################################---MIOT---####################################################################
    M= nx.Graph()
    M.add_node(1)
    M.nodes[1]['Type'] = 'CORE'
    M.nodes[1]['vnf'] = 'AMF'
    M.nodes[1]['cpu'] = 5 
    M.add_node(2)
    M.nodes[2]['Type'] = 'CORE'
    M.nodes[2]['vnf'] = 'AMF'
    M.nodes[2]['cpu'] = 5 
    M.add_node(4)
    M.nodes[4]['Type'] = 'CORE'
    M.nodes[4]['vnf'] = 'AMF'
    M.nodes[4]['cpu'] = 5 
    M.add_node(3)
    M.nodes[3]['Type'] = 'CORE'
    M.nodes[3]['vnf'] = 'SMF'
    M.nodes[3]['cpu'] = 5 
    M.add_node(5)
    M.nodes[5]['Type'] = 'EDGE'
    M.nodes[5]['vnf'] = 'UPF'
    M.nodes[5]['cpu'] = 5
    
    M.add_edge(1,3)
    M.edges[(1,3)]['bw'] = 2  
    M.add_edge(2,3)
    M.edges[(2,3)]['bw'] = 2
    M.add_edge(4,3)
    M.edges[(4,3)]['bw'] = 2
    M.add_edge(3,5)
    M.edges[(3,5)]['bw'] = 2
    
    ######################################################################################
    
    RM = {'G':M,'cpu':5,'bw':1,'lat':1,'Type':3}#,'Time':2}
    RB1 = {'G':B,'cpu':8,'bw':4,'lat':0.1,'Type':1}#,'Time':2}
    RB2 = {'G':B,'cpu':5,'bw':3,'lat':0.3,'Type':2}#,'Time':2}
    RE = {'G':E,'cpu':5,'bw':2,'lat':0.8,'Type':4}#,'Time':2}
    
    
    
    
    
    
    