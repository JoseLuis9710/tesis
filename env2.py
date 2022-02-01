# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:13:40 2020

@author: Santiago Matinez
"""


import operator
import random 
#from random import sample
import numpy as np
import networkx as nx 
#import matplotlib.pyplot as plt
from operator import itemgetter
from request import request as req
#import multiprocessing
import time

class environment:
    
    def __init__(self,nod,con):
        #np.random.seed(seed=2)
        self.conteo = {1:0,2:0,3:0,4:0}
        self.conteo_total = {1:0,2:0,3:0,4:0}
        self.my_slices = {1:list(),2:list(),3:list(),4:list()}
        self.disponibles = {1:list(),2:list(),3:list(),4:list()}
        self.wg = []
        self.slices = list()
        nodes = nod
        conexiones = con
        n_cores= int(nodes*0.25)
        n_edges= int(nodes*0.75)
        self.G = nx.barabasi_albert_graph(nodes,conexiones,seed=2)
        maximos = list()
        vecinos = [len(list(self.G.neighbors(n))) for n in self.G.nodes()]
        for i in range(0,n_cores):
            valor_max=max(vecinos)
            maximo = vecinos.index(valor_max)
            maximos.append(maximo)
            vecinos[maximo]=0
        self.core_nodes=maximos
        self.time_units = 0
        self.R = list()
        self.R.append(req.RB1)
        self.R.append(req.RB2)
        self.R.append(req.RM)
        self.R.append(req.RE)
        self.create_actions()
        self.edge_nodes = list(self.G.nodes() - self.core_nodes)
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, nodelist=self.core_nodes, node_color="r",with_labels=True)
        nx.draw(self.G, pos, nodelist=self.edge_nodes, node_color="b",with_labels=True)
        self.estados = list()
        pp = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        for psr1 in pp:
            for psr2 in pp:
                for pmiot in pp:
                    for pemb in pp:
                        self.estados.append([psr1,psr2,pmiot,pemb])
        self.llenar()
        self.buscar()
        
        
    def llenar_cpu(self):
        for node in self.edge_nodes:  
            self.G.nodes[node]['cpu'] = 100
            self.G.nodes[node]['Type'] = 'EDGE'

        for node in self.core_nodes:
            self.G.nodes[node]['cpu'] = 400
            self.G.nodes[node]['Type'] = 'CORE'
            
    def llenar_bw(self):
        for link in self.G.edges:
            self.G.edges[link]['bw'] = 100 
        
    def llenar_latencia(self):
        for link in self.G.edges():
            if (link[0] in self.edge_nodes) and (link[1] in self.edge_nodes):
                self.G.edges[link]['lat'] = random.uniform(0.01,0.3)
            elif (link[0] in self.core_nodes) and (link[1] in self.core_nodes):
                self.G.edges[link]['lat'] = random.uniform(0.01,0.3)
            else: 
                self.G.edges[link]['lat'] = random.uniform(0.01,1)
        
    def llenar(self):
        self.llenar_cpu()
        self.llenar_bw()
        self.llenar_latencia()
        
    def get_EP(self,nodes,RSR):
        cpu = RSR['cpu']
        rank = dict()
        for node in nodes:
            if self.G.nodes[node]['cpu'] >= cpu:
                BW = 0
                LAT = 0
                for link in self.G.edges(node):
                    BW = BW + self.G.edges[link]['bw']
                    LAT = LAT + self.G.edges[link]['lat']
                LAT /= len(self.G.edges(node))
                BW /= len(self.G.edges(node))
                EP = self.G.nodes[node]['cpu'] * BW * 1/LAT
                rank[node] = EP
        rank = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)
        return rank
    
    def mostrar(self):
    
        print("LINKS: ")
        for link in self.G.edges():
            print(link,self.G.edges[link]['bw'],self.G.edges[link]['lat'])
       
        print("NODES: ")
        for node in self.G.nodes():
            print(node,self.G.nodes[node]['cpu'])
         
    def comparar(self,vecinos_red,vecinos_grafo,RSR):
        C = RSR['G']
        nodos = list()
        rank_nodes = list()
        rank = self.get_EP(vecinos_red,RSR)
        for (vec,value) in rank:
            rank_nodes.append(vec)
        band = 'SI'
        
        for nodo_grafo in vecinos_grafo:
            band2 = 1
            for nodo_red in rank_nodes:
                if (C.nodes[nodo_grafo]['Type'] ==  self.G.nodes[nodo_red]['Type']) and (band2 == 1):
                    nodos.append(nodo_red)
                    rank_nodes.remove(nodo_red)
                    band2 = 0

        if len(nodos) < len(vecinos_grafo):
            band = 'NO'
        elif RSR['Type'] in [1,2] and len(vecinos_grafo)>2:
            if (nodos[-2],nodos[-1]) in self.G.edges() and C.nodes[vecinos_grafo[-2]]['vnf'] == 'UPF-BK' and C.nodes[vecinos_grafo[-1]]['vnf'] == 'UPF-BK':
            #if (nodos) in G.edges() and C.nodes[vecinos_grafo[-2]]['vnf'] == 'UPF-BK' and C.nodes[vecinos_grafo[-1]]['vnf'] == 'UPF-BK':
                band='SI'
            else:
                band='NO'
        return band,nodos
    
#############################################################################################
    def potencia(self,c):
        """Calcula y devuelve el conjunto potencia del 
           conjunto c.
        """
        if len(c) == 0:
            return [[]]
        r = self.potencia(c[:-1])
        return r + [s + [c[-1]] for s in r]

    def combinaciones(self,c, n):
        """Calcula y devuelve una lista con todas las
           combinaciones posibles que se pueden hacer
           con los elementos contenidos en c tomando n
           elementos a la vez.
        """
        return [s for s in self.potencia(c) if len(s) == n]
    
    def inserta(self,x, lst, i):
        """Devuelve una nueva lista resultado de insertar
           x dentro de lst en la posición i.
        """
        return lst[:i] + [x] + lst[i:]
    
    def inserta_multiple(self,x, lst):
        """Devuelve una lista con el resultado de
           insertar x en todas las posiciones de lst.  
        """
        return [self.inserta(x, lst, i) for i in range(len(lst) + 1)]
    
    def permuta(self,c):
        """Calcula y devuelve una lista con todas las
           permutaciones posibles que se pueden hacer
           con los elementos contenidos en c
        """
        from tqdm import tqdm
        if len(c) == 0:
            return [[]]
        return sum([self.inserta_multiple(c[0], s)
                    for s in self.permuta(c[1:])],
                   [])
    
    def verificar(self,path,RSR):
        lat=0
        grafo = RSR['G']
        
        for virtual_link in grafo.edges():
            node_1 = path[virtual_link[0]]
            node_2 = path[virtual_link[1]]
            if self.G.edges[(node_1,node_2)]['bw'] >= RSR['bw'] :
                if RSR['Type'] in [1,2] :
                    if grafo.nodes[virtual_link[0]]['vnf'][-2:] != 'BK' and grafo.nodes[virtual_link[1]]['vnf'][-2:] != 'BK' :
                        lat = lat + self.G.edges[(node_1,node_2)]['lat']
                else:
                    lat = lat + self.G.edges[(node_1,node_2)]['lat']
            else:
                return 'NO',lat

        if path not in self.my_slices[RSR['Type']]:
            self.conteo_total[RSR['Type']] = self.conteo_total[RSR['Type']] + 1
            return 'SI',lat
        return 'NO',lat
    
    def verificar_lat(self,path,RSR):
        lat=0
        grafo = RSR['G']
        cpu = RSR['cpu']
        
        for virtual_link in grafo.edges():
            node_1 = path[virtual_link[0]]
            node_2 = path[virtual_link[1]]
            if self.G.edges[(node_1,node_2)]['bw'] >= RSR['bw'] :
                if RSR['Type'] in [1,2] :
                    if grafo.nodes[virtual_link[0]]['vnf'][-2:] != 'BK' and grafo.nodes[virtual_link[1]]['vnf'][-2:] != 'BK' :
                        lat = lat + self.G.edges[(node_1,node_2)]['lat']
                else:
                    lat = lat + self.G.edges[(node_1,node_2)]['lat']
            else:
                return 'NO',lat
            
        for node in path.values():
            if self.G.nodes[node]['cpu'] < cpu:
                return 'NO',lat

        if lat <= RSR['lat']:
            self.conteo[RSR['Type']] = self.conteo[RSR['Type']] + 1
            return 'SI',lat
        return 'NO',lat
        
    def get_path(self,RSR):
        grafo = RSR['G']
        deploys = dict()
        rank_edges = self.get_EP(self.edge_nodes,RSR)
        rank_cores = self.get_EP(self.core_nodes,RSR)
        contador_vec_upf=0
        contador_vec_smf=0
        contador_vec_amf=0
        if RSR['Type'] in [1,2,3]:
            rank = rank_edges
        else:
            rank = rank_cores
            
        for (upf,value) in rank:
            grafo = RSR['G']
            permutaciones_upf=list()
            vecinos_upf = list(self.G.neighbors(upf))
            for node in grafo.nodes():
                if grafo.nodes[node]['vnf'] == 'UPF': 
                    upf_grafo = node
            deploys[upf_grafo]=upf
            deploys1=deploys
            vecinos_upf_grafo = list(grafo.neighbors(upf_grafo))
            combinaciones_upf=self.combinaciones(vecinos_upf,len(vecinos_upf_grafo))
            for comb in combinaciones_upf:
                permutaciones_upf = permutaciones_upf + self.permuta(comb)
            for permupf in permutaciones_upf:
                band,vecinos_upf=self.comparar(permupf,vecinos_upf_grafo,RSR)
                contador_vec_upf+=1
                if band =='SI':
                    permutaciones_smf=list()
                    for vec_grafo,vec_red in zip(vecinos_upf_grafo,vecinos_upf):
                        deploys[vec_grafo]=vec_red
                    deploys2=deploys
                    if RSR['Type'] in [1,2]:
                        smf=deploys[2]
                        smf_grafo=2
                    if RSR['Type'] in [3,4]:
                        smf=deploys[3]
                        smf_grafo=3
                    vecinos_smf_grafo = list(grafo.neighbors(smf_grafo))
                    vecinos_smf = list(self.G.neighbors(smf))
                    for vec in list(deploys.values()):
                        if vec in vecinos_smf:
                            vecinos_smf.remove(vec)
                    for vec in list(deploys.keys()):
                        if vec in vecinos_smf_grafo:
                            vecinos_smf_grafo.remove(vec)
                    combinaciones_smf=self.combinaciones(vecinos_smf,len(vecinos_smf_grafo))
                    for comb in combinaciones_smf:
                        permutaciones_smf = permutaciones_smf + self.permuta(comb)
                    for permsmf in permutaciones_smf:
                        band,vecinos_smf=self.comparar(permsmf,vecinos_smf_grafo,RSR)
                        contador_vec_smf+=1
                        if band =='SI':
                            permutaciones_amf=list()
                            for vec_grafo,vec_red in zip(vecinos_smf_grafo,vecinos_smf):
                                deploys[vec_grafo]=vec_red
                            deploys3=deploys
                            if RSR['Type'] in [1,2]:
                                amf=deploys[1]
                                amf_grafo=1
                                vecinos_amf_grafo = list(grafo.neighbors(amf_grafo))
                                vecinos_amf = list(self.G.neighbors(amf))
                                for vec in list(deploys.values()):
                                    if vec in vecinos_amf:
                                        vecinos_amf.remove(vec)
                                for vec in list(deploys.keys()):
                                    if vec in vecinos_amf_grafo:
                                        vecinos_amf_grafo.remove(vec)
                                combinaciones_amf=self.combinaciones(vecinos_amf,len(vecinos_amf_grafo))
                                for comb in combinaciones_amf:
                                    permutaciones_amf = permutaciones_amf + self.permuta(comb)
                                
                                for permamf in permutaciones_amf:
                                    band,vecinos_amf=self.comparar(permamf,vecinos_amf_grafo,RSR)
                                    contador_vec_amf+=1
                                    if band =='SI':
                                        for vec_grafo,vec_red in zip(vecinos_amf_grafo,vecinos_amf):
                                            deploys[vec_grafo]=vec_red
                                        deploys4=deploys
                                        band2,lat=self.verificar(deploys,RSR)
                                        
                                        if band2=='SI':
                                            sl = deploys.copy()
                                            self.my_slices[RSR['Type']].append(sl)
                                            
                                            del deploys[6]
                                            continue
                                        else:
                                            del deploys[6]
                                            
                                            continue
                                    else:
                                        continue
                                del deploys[1]
                                del deploys[7]
                                contador_vec_amf=0
                            else:
                                band3,lat=self.verificar(deploys,RSR)
                                if band3=='SI':
                                    sl = deploys.copy()
                                    self.my_slices[RSR['Type']].append(sl)
                                    del deploys[1]
                                    del deploys[2]
                                    if RSR['Type'] in [3]:
                                        del deploys[4]
                                    contador_vec_amf=0
                                    continue
                                else:
                                    del deploys[1]
                                    del deploys[2]
                                    if RSR['Type'] in [3]:
                                        del deploys[4]
                                    contador_vec_amf=0
                                    continue
                            
                        else:
                            continue
                    if RSR['Type'] in [1,2]:
                        del deploys[2]
                        del deploys[4]
                        del deploys[5]
                        contador_vec_smf=0
                    
                    if RSR['Type'] in [3,4]:
                        del deploys[3]
                        contador_vec_smf=0
                    
                else:
                    continue
    
    def deploy(self,RSR):
        grafo = RSR['G']
        if len(self.disponibles[RSR['Type']]) != 0:
            for sl in self.disponibles[RSR['Type']]:
                band = 1
                for node in sl[0].values():
                    if self.G.nodes[node]['cpu'] - RSR['cpu'] < 0:
                        band = 0
                        break
                for virtual_link in grafo.edges():
                    node_1 = sl[0][virtual_link[0]]
                    node_2 = sl[0][virtual_link[1]]
                    if self.G.edges[(node_1,node_2)]['bw'] - RSR['bw'] < 0:
                        band = 0
                        break
                        
                if band == 1:
                    
                    for node in sl[0].values():
                        self.G.nodes[node]['cpu'] = self.G.nodes[node]['cpu'] - RSR['cpu']

                    for virtual_link in grafo.edges():
                        node_1 = sl[0][virtual_link[0]]
                        node_2 = sl[0][virtual_link[1]]
                        self.G.edges[(node_1,node_2)]['bw'] = self.G.edges[(node_1,node_2)]['bw'] - RSR['bw']

                    self.slices.append(((sl[1],sl[0]),RSR))
                    return 'SI',sl[0],sl[1]
                
                else:
                    continue
                
            return 'NO',sl[0],sl[1]
        
        else:
            if RSR['Type'] in [1,2]:
                return 'NO',{3: 10, 2: 14, 4: 7, 5: 8, 1: 22, 7: 13, 6: 11}, 0.7
            if RSR['Type'] in [3]:
                return 'NO',{5: 20, 3: 13, 1: 4, 2: 2, 4: 7}, 1.2
            if RSR['Type'] in [4]:
                return 'NO',{4: 13, 3: 7, 1: 5, 2: 8}, 1
        
    def cobrar(self,path,RSR):
        lat = path[0]
        nodes = path[1].values()
        cost = 0
        
        if False:#RSR['Type'] == 1:
            for node in nodes:
                if node in self.edge_nodes:
                    cost = cost + RSR['cpu'] * 6
                if node in self.core_nodes:
                    cost = cost + RSR['cpu'] * 4
            cost = cost + len(RSR['G'].edges()) * RSR['bw'] * 2
        else:
            for node in nodes:
                if node in self.edge_nodes:
                    cost = cost + RSR['cpu'] * 3
                if node in self.core_nodes:
                    cost = cost + RSR['cpu'] * 2
            cost = cost + len(RSR['G'].edges()) * RSR['bw'] * 1
        return cost
        
    
    def liberar(self,sli):
        R = sli[1]
        path = sli[0]
        grafo = R['G']
        for node in path[1].values():
            self.G.nodes[node]['cpu'] = self.G.nodes[node]['cpu'] + R['cpu']
        
        for virtual_link in grafo.edges():
            node_1 = path[1][virtual_link[0]]
            node_2 = path[1][virtual_link[1]]
            self.G.edges[(node_1,node_2)]['bw'] = self.G.edges[(node_1,node_2)]['bw'] + R['bw']
            
        indice = self.slices.index((path,R))
        self.slices.pop(indice) 
        
    def check_time(self):
        self.time_units = self.time_units + 1
        for pet in self.slices:
            pet[1]['Time'] -= 1
            if pet[1]['Time'] < 0:
                self.liberar(pet)
                
    def buscar(self):
        for i,p in enumerate(self.R):
            self.get_path(p)
    
    def revisar(self):
        #t1=time.time()
        for tipo in self.my_slices.keys():
            for sl in self.my_slices[tipo]:
                band,lat = self.verificar_lat(sl,self.R[tipo-1])
                #print(tipo,sl)
                if band == 'SI':
                    self.disponibles[tipo].append((sl,lat))
        #t2 = time.time()
        #print('TIEMPO DE VERIFICACIÓN LATENCIA:',t2-t1)
    def window_time(self,surgery):
        
        if surgery == False:
            tam = 4
            lam = 5
        else:
            tam = 2
            lam = 40
            
        self.pet_llegadas = [0 for _ in range(0,tam)]
        
        for i in range(0,2):
            self.pet_llegadas = self.pet_llegadas + np.random.poisson(lam,size=tam)
            self.check_time()
        return self.pet_llegadas
    
    def monitoreo(self):
        
        nodos = {1:list(),2:list(),3:list(),4:list()}
        links = {1:list(),2:list(),3:list(),4:list()}
        potenciales = np.array([0,0,0,0],dtype=float)
        bw = np.array([0,0,0,0],dtype=float)
        cpu_edge = np.array([0,0,0,0],dtype=float)
        cpu_core = np.array([0,0,0,0],dtype=float)
        
        for tipo in self.disponibles.keys():
            for sl in self.disponibles[tipo]:
                for node in sl[0].values():
                    if node not in nodos[tipo]:
                        nodos[tipo].append(node)
                        
        for tipo in self.disponibles.keys():
            for sl in self.disponibles[tipo]:
                for virtual_link in self.R[tipo-1]['G'].edges():
                    node_1 = sl[0][virtual_link[0]]
                    node_2 = sl[0][virtual_link[1]]
                    if (node_1,node_2) not in links[tipo] and (node_2,node_1) not in links[tipo]:
                        links[tipo].append((node_1,node_2))
                        
        for tipo in self.disponibles.keys():
            conteo_edge = list()
            conteo_core = list()
            conteo_bw = list()
            
            if len(nodos[tipo]) !=0:
                for nodo in nodos[tipo]:
                    if nodo in self.edge_nodes:
                        conteo_edge.append(self.G.nodes[nodo]['cpu'] /100)
                    else:
                        conteo_core.append(self.G.nodes[nodo]['cpu'] /400)
                for link in links[tipo]:
                    conteo_bw.append(self.G.edges[link]['bw']/100)
                
                if tipo != 4:
                    cpu_edge[tipo-1] = sum(conteo_edge)/len(conteo_edge)
                #print(conteo_core,sum(conteo_core),sum(conteo_core)/len(conteo_core))
                cpu_core[tipo-1] = sum(conteo_core)/len(conteo_core)
                #print(cpu_core[tipo-1])
                #time.sleep(3)
                #cpu_core[tipo-1] = sum(conteo_core)/len(conteo_core)
                bw[tipo-1] = sum(conteo_bw)/len(conteo_bw)
            else:
                continue
        potenciales[0] = 0.25 * cpu_edge[0] + 0.5 * cpu_core[0] + 0.25 * bw[0]
        potenciales[1] = 0.25 * cpu_edge[1] + 0.5 * cpu_core[1] + 0.25 * bw[1]
        potenciales[2] = 0.25 * cpu_edge[2] + 0.5 * cpu_core[2] + 0.25 * bw[2]
        #potenciales[3] = 0.5 * cpu_core[3] + 0.5 * bw[3]
        #potenciales[0] = 0.15 * cpu_edge[0] + 0.75 * cpu_core[0] + 0.1 * bw[0]
        #potenciales[1] = 0.15 * cpu_edge[1] + 0.75 * cpu_core[1] + 0.1 * bw[1]
        #potenciales[2] = 0.1 * cpu_edge[2] + 0.65 * cpu_core[2] + 0.25 * bw[2]
        potenciales[3] = 0.7 * cpu_core[3] + 0.3 * bw[3]
        self.ga=potenciales[0]
        self.gb=potenciales[1]
        self.gc=potenciales[2]
        self.gd=potenciales[3]
        
        #print('POTENCIALES',list(potenciales*10)+list(self.pet_llegadas))
        return list(potenciales*10) + list(self.pet_llegadas)
    
    def maximo_cobro(self):
        
        tipos = [0,0,0,0]
        nodos = []#{1:list(),2:list(),3:list(),4:list()}
        links = []#{1:list(),2:list(),3:list(),4:list()}
        cpu = 0
        bw = 0
        
        
        for tipo in self.disponibles.keys():
            for sl in self.disponibles[tipo]:
                for node in sl[0].values():
                    if node not in nodos:
                        nodos.append(node)
                        if node in self.edge_nodes:
                            cpu = cpu + 100 * 3
                            #tipos[tipo-1] = tipos[tipo-1] + self.G.nodes[node]['cpu'] * 3
                        if node in self.core_nodes:
                            cpu = cpu + 400 * 2
                            #tipos[tipo-1] = tipos[tipo-1] + self.G.nodes[node]['cpu'] * 2
                        
        for tipo in self.disponibles.keys():
            for sl in self.disponibles[tipo]:
                for virtual_link in self.R[tipo-1]['G'].edges():
                    node_1 = sl[0][virtual_link[0]]
                    node_2 = sl[0][virtual_link[1]]
                    if (node_1,node_2) not in links and (node_2,node_1) not in links:
                        bw = bw + 100 
                        #tipos[tipo-1] = tipos[tipo-1] + self.G.edges[(node_1,node_2)]['bw']
                        links.append((node_1,node_2))
        #print(nodos)
        return cpu+bw

    def env_start(self,surgery=False):
        self.conteo = {1:0,2:0,3:0,4:0}
        self.disponibles = {1:list(),2:list(),3:list(),4:list()}
        self.slices=list()
        self.llenar()
        self.revisar()
        self.window_time(surgery)
        self.acept = 0
        self.dsp = 0
        self.arriv=0
        self.cir = 0
        self.cir_acep = 0
        self.cir2 = 0
        self.cir2_acep = 0
        self.emb = 0
        self.emb_acep = 0
        self.miot = 0
        self.miot_acep = 0
        self.sum_profit = 0
        self.step_mon = []
        self.step_mon_emb = []
        self.step_mon_miot = []
        self.step_mon2 = []
        st_red=self.monitoreo()
        #self.CPU_GLOBAL = 0
        #self.BW_GLOBAL = 0
        #print(st_red)
        self.conteo_cir = []
        self.conteo_cir2 = []
        self.multas = 0
        
        return st_red,False
        
    def priorizador(self,action):
        
        slices = list()
        pesos = self.actions[action]
        pets = np.array([pesos]) * self.pet_llegadas
        #print('llegadas',self.pet_llegadas)
        
        pets = pets.astype(int)[0]
        #print('ACEPTADOS:',pets)
        #print('definitivo',pets)
        self.cir +=pets[0]
        self.cir2 +=pets[1]
        self.miot +=pets[2]
        self.emb +=pets[3]
        self.cir_step =pets[0]
        self.cir2_step =pets[1]
        self.miot_step =pets[2]
        self.emb_step =pets[3]
        self.cir2_step_p =pesos[1]
        self.miot_step_p =pesos[2]
        self.emb_step_p =pesos[3]
        #print('SR1 ACEPTADAS:',pets[0])
        #print('SR2 ACEPTADAS:',pets[1])
        #print('MIOT ACEPTADAS:',pets[2])
        #print('EMBB ACEPTADAS:',pets[3])
        #print(pets)
        for i,n in enumerate(pets):
            peticiones = list()
            for _ in range(0,n):
                
                if i + 1 == 1 or i + 1 == 2:
                    time = np.random.poisson(30)#30
                if i + 1 == 3:
                    time = np.random.poisson(8)#8
                if i + 1 == 4:
                    time = np.random.poisson(15)#15
                if time <= 0:
                    time = 2
                p = self.R[i].copy()
                p['Time'] = time
                peticiones.append(p)
            ordenadas = sorted(peticiones, key=itemgetter('Time'),reverse=True)
            slices = slices + ordenadas
        return slices
        
    def env_step(self,action,surgery=False):
        #t1=time.time()
        prof = {1:0,2:0,3:0,4:0}
        proff = 0
        my_dsp =0
        self.cir_acep_step = 0
        self.cir2_acep_step = 0
        self.emb_acep_step = 0
        self.miot_acep_step = 0
        self.profit_bueno=[0,0,0,0]
        self.profit_total=[0,0,0,0]
        #self.CPU,self.BW = self.maximo_cobro()
        maxx = self.maximo_cobro()
        #print(maxx)
        #print('ENTRANTES:',sum(self.pet_llegadas))
        self.arriv+=sum(self.pet_llegadas)
        aceptados = self.priorizador(action)
        self.acept += len(aceptados)
        reward_pro = 0
        latencias=list()
        for RSR in aceptados:
            band,path,lat = self.deploy(RSR)
            #print(band,path)
            #for no_co in self.edge_nodes:
                #print(no_co,self.G.nodes[no_co]['cpu'])
            if band == 'SI':
                my_dsp += 1 
                self.dsp += 1
                #desplegados.append([path,RSR])
                reward_pro += self.cobrar((lat,path),RSR) * RSR['Time']
                #self.resources((lat,path),RSR)
                #proff = proff + self.cobrar((lat,path),RSR)
                self.profit_bueno[RSR['Type']-1]+=self.cobrar((lat,path),RSR) * RSR['Time']
                self.profit_total[RSR['Type']-1]+=self.cobrar((lat,path),RSR) * RSR['Time']
                latencias.append((RSR['Type'],lat))

                if RSR['Type'] == 1:
                    self.cir_acep+=1
                    self.cir_acep_step +=1
                if RSR['Type'] == 2:
                    self.cir2_acep+=1
                    self.cir2_acep_step +=1
                if RSR['Type'] == 3:
                    self.miot_acep+=1
                    self.miot_acep_step +=1
                if RSR['Type'] == 4:
                    self.emb_acep+=1
                    self.emb_acep_step +=1
            else:
                reward_pro -= self.cobrar((lat,path),RSR) * RSR['Time']
                self.multas += self.cobrar((lat,path),RSR) * RSR['Time']
                self.profit_total[RSR['Type']-1]+=self.cobrar((lat,path),RSR) * RSR['Time']
                latencias.append((RSR['Type'],lat))
                
        #contador = 0
        #bb = 0
        #for i in [1,2,3,4]:
            #if self.CPU[i]+self.BW[i] != 0:
                #bb += prof[i]/(self.CPU[i]+self.BW[i])
            #else:
                #bb += 1
        #print(proff)
        #time.sleep(5)
        #print('DESPLEGADOS:',self.cir_acep_step,self.cir2_acep_step,self.miot_acep_step,self.emb_acep_step)
        self.sum_profit += reward_pro
        reward_late=self.reward_lat(latencias)
        
        #self.U = (self.CPU_GLOBAL/self.maximo_cobro()[0] + self.BW_GLOBAL/self.maximo_cobro()[1]) / 2
        
        ############################## CIRUGIA 
        if self.cir_step == 0 and self.ga == 0:
            xx=1
        elif self.cir_step == 0 and self.ga != 0:
            xx =0
        else:
            xx = self.cir_acep_step/self.cir_step
            self.conteo_cir.append(self.cir_acep_step/self.cir_step)
        self.step_mon.append(xx)
        ############################## CIRUGIA 2
        # if self.cir2_step_p == 0 and self.gb == 0:
        #     rr=1
        # elif self.cir2_step_p == 0 and self.gb != 0:
        #     rr=0
        # else:
        #     if self.cir2_step != 0:
        #         rr = self.cir2_acep_step/self.cir2_step
        #     else:
        #         rr = 1
        # self.step_mon2.append(rr)
        if self.cir2_step_p == 0 and self.gb == 0:
            rr=1
        else:
            if self.cir2_step != 0:
                rr = self.cir2_acep_step/self.cir2_step
                self.conteo_cir2.append(rr)
            else:
                rr = 1
        self.step_mon2.append(rr)
        ############################### MIOT
        # if self.miot_step_p == 0 and self.gc == 0:
        #     zz=1
        # elif self.miot_step_p == 0 and self.gc != 0:
        #     zz=0
        # else:
        #     if self.miot_step != 0:
        #         zz = self.miot_acep_step/self.miot_step
        #     else:
        #         zz = 1
        # self.step_mon_miot.append(zz)
        if self.miot_step_p == 0 and self.gc == 0:
            zz=1
        else:
            if self.miot_step != 0:
                zz = self.miot_acep_step/self.miot_step
            else:
                zz = 1
        self.step_mon_miot.append(zz)
        ############################### EMBB
        # if self.emb_step_p == 0 and self.gd == 0:
        #     yy=1
        # elif self.emb_step_p == 0 and self.gd != 0:
        #     yy=0
        # else:
        #     if self.emb_step != 0:
        #         yy = self.emb_acep_step/self.emb_step
        #     else:
        #         yy = 1
        # self.step_mon_emb.append(yy)
        if self.emb_step_p == 0 and self.gd == 0:
            yy=1
        else:
            if self.emb_step != 0:
                yy = self.emb_acep_step/self.emb_step
            else:
                yy = 1
        self.step_mon_emb.append(yy)
        
        p1=(2*self.profit_bueno[0]-self.profit_total[0])/(maxx*40) if maxx  !=0 else 0#self.profit_total[0] if self.profit_total[0]  !=0 else 0
        p2=(2*self.profit_bueno[1]-self.profit_total[1])/(maxx*40) if maxx  !=0 else 0#self.profit_total[1] if self.profit_total[1]  !=0 else 0
        p3=(2*self.profit_bueno[2]-self.profit_total[2])/(maxx*40) if maxx  !=0 else 0##self.profit_total[2] if self.profit_total[2]  !=0 else 0
        p4=(2*self.profit_bueno[3]-self.profit_total[3])/(maxx*40) if maxx  !=0 else 0#self.profit_total[3] if self.profit_total[3]  !=0 else 0
        #p1=(self.profit_bueno[0])/(maxx*40) if maxx  !=0 else 0#self.profit_total[0] if self.profit_total[0]  !=0 else 0
        #p2=(self.profit_bueno[1])/(maxx*40) if maxx  !=0 else 0#self.profit_total[1] if self.profit_total[1]  !=0 else 0
        #p3=(self.profit_bueno[2])/(maxx*40) if maxx  !=0 else 0##self.profit_total[2] if self.profit_total[2]  !=0 else 0
        #p4=(self.profit_bueno[3])/(maxx*40) if maxx  !=0 else 0#self.profit_total[3] if self.profit_total[3]  !=0 else 0
        
        #print('CIRUGIA',p1)
        #reward_pro = 0.5 * p1 + 0.3 * p2 + 0.1 * p3 + 0.1 * p4
        reward_pro =( p1 + p2 + p3 + p4) * 10
        #print('R_TOTAL',reward_pro)
        reward_total= reward_late * 0.8 + reward_pro * 0.2
        
        self.llenar_latencia()
        self.window_time(surgery)
        self.conteo = {1:0,2:0,3:0,4:0}
        self.disponibles = {1:list(),2:list(),3:list(),4:list()}
        self.revisar()
        st_red = self.monitoreo()
        #t2 = time.time()
        #print('TIEMPO DEL STEP:',t2-t1)
        return reward_total, st_red,False
    
    def total(self):
        suma = 0
        
        for link in self.G.edges():
            suma += self.G.edges[link]['bw'] 
       
        for node in self.G.nodes():
            if node in self.edge_nodes:
                suma += self.G.nodes[node]['cpu'] * 3
            else:
                suma += self.G.nodes[node]['cpu'] * 2
        
        print(suma)
    
    def reward_lat(self,latencias):
        tipos = {1:list(),2:list(),3:list(),4:list()}
        limite=[0.1,0.5,1,0.8]
        y = 705
        for lat in latencias:
            sumat = limite[lat[0]-1] - float(str(lat[1])[:4])
            rt=(np.exp(y*sumat) - np.exp(-y*sumat)) / (np.exp(y*sumat) + np.exp(-y*sumat))
            tipos[lat[0]].append(rt)
            #print(rt)
            #print(lat)
        #print(tipos[1],tipos[2],tipos[3],tipos[4])
        r1=sum(tipos[1])/len(tipos[1]) if len(tipos[1]) !=0 else 0
        r2=sum(tipos[2])/len(tipos[2]) if len(tipos[2]) !=0 else 0
        r3=sum(tipos[3])/len(tipos[3]) if len(tipos[3]) !=0 else 0
        r4=sum(tipos[4])/len(tipos[4]) if len(tipos[4]) !=0 else 0
        sumatotal = 0.4 * r1 + 0.4 * r2 + 0.1 * r3 + 0.1 * r4
        return sumatotal
    
    def create_actions(self):
        
        t1 = [0,0.2,0.5,1]
        t2 = [0,0.3,0.5,1]
        t3 = [0.5,0.7,1]
        t4 = [0.6,0.8,1]
        self.actions = list()
        for b in t1:
             for c in t2:
                 for d in t3:
                     for e in t4:
                         self.actions.append([b,c,d,e])
                        
        #self.actions = [[0.2,0.4,0.7,0.7],[0.2,0.3,0.7,0.5],[0.2,0.3,0.7,0.4],[0.2,0.4,0.7,0.4],[0,0.4,0.7,0.7],[0,0.3,0.7,0.5],[0,0.3,0.7,0.4],[0,0.4,0.7,0.4],[0.2,0.4,0.7,0.4],[0,0.4,0.7,0],[0.2,0.4,0.7,0],[0,0.3,0.2,0.4],[0,0.4,0.5,0.5],[0.2,0.3,0.2,0.4],[0,0,0,0],[0,0,0.8,0.8],[0.5,0.4,1,0.7],[0.5,0.4,1,0.4],[0.2,0.4,1,0.4],[0.5,1,1,0.7],[0.2,1,1,0.8],[0,1,1,0],[0,1,1,0.4],[0,0,1,0.8],[0,0.7,1,0.8],[0,0,1,0.3],[0.1,1,1,0.5],[0,0.2,0.2,0.4],[0.2,0.6,1,1],[0,0,0.2,0.4],[0.2,0,0.7,0.4],[1,1,1,1]]#,[0.9,0.9,0.9,0.9],[1,0,0,0],[1,1,0,0],[0.9,0,0,0],[0.7,0.9,0,0],[1,0,1,1],[0.9,0,0.9,0.9],[0.8,0.9,1,1],[0.8,1,1,1],[0.8,0,0,0]]
        self.actions = [[0.2,0.3,0.7,0.7],[0.2,0.2,0.7,0.5],[0.2,0.2,0.7,0.2],[0.2,0.3,0.7,0.2],[0,0.3,0.7,0.7],[0,0.2,0.7,0.5],[0,0.2,0.7,0.2],[0,0.3,0.7,0.2],[0.2,0.3,0.7,0.2],[0,0.3,0.7,0],[0.2,0.3,0.7,0],[0,0.2,0.2,0.2],[0,0.3,0.5,0.5],[0.2,0.2,0.2,0.2],[0,0,0,0],[0,0,0.8,0.8],[0.5,0.3,1,0.7],[0.5,0.3,1,0.2],[0.2,0.3,1,0.2],[0.5,1,1,0.7],[0.2,1,1,1],[0,1,1,0],[0,1,1,0.2],[0,0,1,1],[0,0.6,1,1],[0,0,1,0.1],[0.1,1,1,0.5],[0,0.1,0.2,0.2],[0.2,0.5,1,1],[0,0,0.2,0.2],[0.2,0,0.7,0.2],[1,1,1,1]]#,[0.9,0.9,0.9,0.9],[1,0,0,0],[1,1,0,0],[0.9,0,0,0],[0.7,0.9,0,0],[1,0,1,1],[0.9,0,0.9,0.9],[0.8,0.9,1,1],[0.8,1,1,1],[0.8,0,0,0]]
        #self.actions = [[0, 0.2, 0.7, 0.2],[0, 1, 1, 0],[0, 0, 1, 0.1],[0.2, 0.5, 1, 1],[0.2, 0, 0.7, 0.2]]

    def rango(self,item):
        
        if item == 0:
            return 0
        if item > 0 and item <= 0.1:
            return 0.1
        if item > 0.1 and item <= 0.2:
            return 0.2
        if item > 0.2 and item <= 0.3:
            return 0.3
        if item > 0.3 and item <= 0.4:
            return 0.4
        if item > 0.4 and item <= 0.5:
            return 0.5
        if item > 0.5 and item <= 0.6:
            return 0.6
        if item > 0.6 and item <= 0.7:
            return 0.7
        if item > 0.7 and item <= 0.8:
            return 0.8
        if item > 0.8 and item <= 0.9:
            return 0.9
        if item > 0.9 and item <= 1:
            return 1
        
    def rango2(self,item):
        
        if item >= 0 and item < 0.1:
            return 0
        if item >= 0.1 and item < 0.2:
            return 0.1
        if item >= 0.2 and item < 0.3:
            return 0.2
        if item >= 0.3 and item < 0.4:
            return 0.3
        if item >= 0.4 and item < 0.5:
            return 0.4
        if item >= 0.5 and item < 0.6:
            return 0.5
        if item >= 0.6 and item < 0.7:
            return 0.6
        if item >= 0.7 and item < 0.8:
            return 0.7
        if item >= 0.8 and item < 0.9:
            return 0.8
        if item >= 0.9 and item < 1:
            return 0.9
        if item == 1:
            return 1
        

