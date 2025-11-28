
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
   
def bow_dataset_cv(K, X_train):
    
    #el que hem de fer és ajuntar tots els descriptors de totes les imatges d'entrenament en una sola matriu (vstack)
    #del X_train agafem només els valors, que són les matrius de descriptors, si no possesim serien els values també
    #els X_train.values és una llista de matrius numpy, cada matriu és (N,128), però nosaltres el que volem és 1 sola matriu numpy(vstack)
    all_train_descriptors = np.vstack(X_train.values) 
    

    # Paràmetres de k-means per OpenCV. criteria és una tupla de 3 elements
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, #indiquem el tipus de les variables per decidir si aturem (iteracions i epsilon)
        100,   # iteracions màximes (per cada kmeans)
        0.01   # tolerància (quan variació sigui menor a tolerància aturem) --> epsilon
    )

    attempts = 10 #Nombre de vegades que fa el kmeans, i es quedarà amb la millor (la suma de distàncies de cada punt al seu centre és la menor possible).
    flags = cv2.KMEANS_PP_CENTERS  # indica com escollim els centres inicials. Enlloc de fer-ho de manera random, PP està dissenyat per triar centres inicials més “separats” i acostuma a donar millors solucions.

    print("Entrenant K-Means amb OpenCV, K =", K, "clusters...")
    # compactness: És un número que indica com de bons són els clusters. És la suma de les distàncies al quadrat de cada descriptor al seu centre assignat. Com més petit, millor.
    # labels: És un array de longitud igual al nombre total de descriptors. Cada element indica a quin cluster ha estat assignat aquest descriptor. Exemple: labels[0] = 3 → el primer descriptor pertany al cluster 3.
    # centers: És la part més important per BoW. És una matriu (K, 128). K = nombre de clusters, 128 = mida de cada descriptor. Cada fila és el centre d’un cluster, és a dir, una “paraula visual” que representa un grup de descriptors similars.
    compactness, labels, centers = cv2.kmeans(
        all_train_descriptors,  # dades (N, 128)
        K,                      # nº de clusters
        None,                   # sense etiquetes inicials
        criteria,               # criteris d'aturada
        attempts,               # intents diferents
        flags                   # com escollir centres inicials
    )

    bow_centers = centers  # shape (K, 128). Tenim K centres i 128 és per la longitud del descriptor. Aquest és el millor cas dels 10 provats.
    print("Shape bow_centers:", bow_centers.shape)

    return bow_centers

def bow_dataset_sklearn(K, X_train):

    #el que hem de fer és ajuntar tots els descriptors de totes les imatges d'entrenament en una sola matriu (vstack)
    #del X_train agafem només els valors, que són les matrius de descriptors, si no possesim serien els values també
    #els X_train.values és una llista de matrius numpy, cada matriu és (N,128), però nosaltres el que volem és 1 sola matriu numpy(vstack)
    all_train_descriptors = np.vstack(X_train.values) 

    # 2. Creem el model KMeans
    model = KMeans(
        init='k-means++',  #com inicialitzar els centres. Evita problemes de centres inicials molt junts i accelera la convergència.
        max_iter=100,      #màxim d’iteracions
        tol=0.01,          #tolerància per parar
        n_init=10,         #intents diferents amb inicialitzacions diferents
        random_state=42,
        n_clusters=K)
    model.fit(all_train_descriptors)
    bow_centers = model.cluster_centers_
    return bow_centers



def bow_dataset2(X_train):

    # 1. Apil·lem tots els descriptors en una sola matriu
    all_train_descriptors = np.vstack(X_train.values)

    # 2. Creem el model KMeans
    model = KMeans(
        init='k-means++',  # com inicialitzar els centres ('random'també és possible)
        max_iter=100,      # màxim d’iteracions
        tol=0.01,          # tolerància per parar
        n_init=10,         # intents diferents amb inicialitzacions diferents
        random_state=42
    )   

    # 3. Creem el visualizer i triem la mètrica elbow o silhouette
    
    # KElbowVisualizer → eina de Yellowbrick que ajuda a triar automàticament K.
    # k=(40, 50) → prova valors de K entre 40 i 50.
    # metric='distortion' → calcula la distorsió, és a dir la suma de distàncies dels descriptors als centres assignats.
    
    visualizer = KElbowVisualizer(model, k=(40, 50), metric='distortion')  # ajusta k màxim segons el que vulguis
    visualizer.fit(all_train_descriptors)

    # 4. Millor nombre de clusters automàtic

    # visualizer.elbow_value_ → retorna el K on l’elbow és més clar.
    # Això ens dóna el nombre ideal de paraules visuals per al nostre dataset.
    
    best_k = visualizer.elbow_value_
    print("Millor k trobat:", best_k)

    # 5. Entrenem KMeans final amb el millor k
    final_kmeans = KMeans(n_clusters=best_k)
    final_kmeans.fit(all_train_descriptors)
    bow_centers = final_kmeans.cluster_centers_
    print("Shape bow_centers:", bow_centers.shape)





