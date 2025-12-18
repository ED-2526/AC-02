
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns




def histogram_intersection_kernel(X, Y):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        K[i, :] = np.sum(np.minimum(x, Y), axis=1)
    return K





def calcula_prediccio_svm_double_step(X_train_df, Y_train_ser, X_test_df, classes_postres, C_primera=0.5, C_segona=0.5, gamma='scale', kernel='rbf', degree=3):

    X_train = X_train_df.values
    X_test = X_test_df.values
    y_train = Y_train_ser.values

    
    kernel_func = histogram_intersection_kernel if kernel == 'hist_intersection' else kernel

    #classificador 1, que és igual que l'anterior
    print("Entrenant Model A ...")
    svm_general = SVC(C=C_primera, class_weight='balanced', gamma=gamma, kernel=kernel_func, degree=degree)
    svm_general.fit(X_train, y_train)
    
    y_pred_final = svm_general.predict(X_test)

    #classificador 2, el especialista
    mask_train_postres = np.isin(y_train, classes_postres) #agafem nomes les imatges de postres del train
    
    if np.sum(mask_train_postres) > 0: #si tenim imatges de postres al train
        
        #aqui el que fem és de totes les imatges ens quedem només les de postres a partir de la mascara que hem creat abans
        X_train_postres = X_train[mask_train_postres]
        y_train_postres = y_train[mask_train_postres]

        #ara entrenem el segon model només amb les imatges de postres
        svm_especialista = SVC(
            C=C_segona, 
            class_weight='balanced', 
            gamma=gamma, 
            kernel=kernel_func, 
            degree=degree, 
            decision_function_shape='ovr'  
        )
        svm_especialista.fit(X_train_postres, y_train_postres)

        # Refinament
        mask_test_detected_as_postres = np.isin(y_pred_final, classes_postres) #mirem quines imatges de prediccio son de postres
        num_revisions = np.sum(mask_test_detected_as_postres) #comptem quantes n'hi ha
        
        if num_revisions > 0: #si n'hi ha alguna
            
            X_to_refine = X_test[mask_test_detected_as_postres] #agafem les imatges de test que han estat classificades com a postres
            y_pred_refined = svm_especialista.predict(X_to_refine) #les passem pel segon model per refinar la prediccio
            y_pred_final[mask_test_detected_as_postres] = y_pred_refined #actualitzem les prediccions finals amb les refinades

    return y_pred_final
