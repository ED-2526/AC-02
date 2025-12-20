
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.cluster import KMeans

def bow_histogram(descriptors, kmeans_model, k): #calcula l'historgama per cada imatge

    #utilitzem el model kmeans ja entrenat per predir la paraula visual de cada descriptor
    words = kmeans_model.predict(descriptors)  
    #constuïm l'histograma comptant quantes vegades apareix cada paraula visual
    #els bins són el nombre de centres, el rang és des de 0 fins al nombre de centres
    #hist és un array de longitud nombre de centres on a cada posició tenim quantes paraules visuals corresponen a aquell centre

    hist = np.bincount(words, minlength=k)
    #normalitzem l'histograma perquè la suma sigui 1, ja que podria ser que en algun cas hi hagués més descriptors que en un altre
    hist = hist / hist.sum()
    return hist

#aquesta funció calcula els histogrames per totes les imatges del conjunt d’entrenament.
#x → llista o pandas.Series amb descriptors per cada imatge.
def calcula_histograma(X, kmeans_model, k):
    

    hists = []
    for desc in X:
        hists.append(bow_histogram(desc, kmeans_model, k))
    return pd.DataFrame(hists, index=X.index) #retornem un dataframe on cada fila és l'histograma d'una foto i cada columna una paraula visual
    #lo del X.index és per mantenir els indexos originals del X d'entrada i després poder-ho emparellar amb les Y

    



