
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd

def bow_histogram(descriptors, centers): #calcula l'historgama per cada imatge
    
    #calculem la distància entre cada descriptor i cada centre, es guarda en una matriu de (N_descriptors, K paraules visuals)
    #és una matriu de distàncies, on de cada fila tenim la distància de cada descriptor a cada centre 
    #per defecte ho fa amb la distància euclidiana, però podem canviar-ho per estudiar el rendiment
    dists = cdist(descriptors, centers)

    #per cada descriptor, triem la paraula visual més propera (centre més proper), canviar a soft assignment per veure si millora 
    #words és un array de longitud N_descriptors, on cada element és l'índex del centre més proper
    words = np.argmin(dists, axis=1)

    #constuïm l'histograma comptant quantes vegades apareix cada paraula visual
    #els bins són el nombre de centres, el rang és des de 0 fins al nombre de centres
    #hist és un array de longitud nombre de centres on a cada posició tenim quantes paraules visuals corresponen a aquell centre
    hist, _ = np.histogram(words, bins=len(centers), range=(0, len(centers)))

    #normalitzem l'histograma perquè la suma sigui 1, ja que podria ser que en algun cas hi hagués més descriptors que en un altre
    #passem a float per evitar problemes de divisió entre enters (pq al dividir passem de enters a floats) i dividim cada element entre la suma total de l'histograma
    hist = hist.astype(float) / np.sum(hist)

    return hist

def calcula_histograma_train(X, centers):
    
    hists = []
    for desc in X:
        hists.append(bow_histogram(desc, centers))
    return pd.DataFrame(hists, index=X.index) #retornem un dataframe on cada fila és l'histograma d'una foto i cada columna una paraula visual
    #lo del X.index és per mantenir els indexos originals del X d'entrada i després poder-ho emparellar amb les Y

    
