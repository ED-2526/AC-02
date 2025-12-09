from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
import pandas as pd



def calcula_prediccio_svm(X_train_df, Y_train_ser, X_test_df, C, gamma='scale', kernel='rbf', degree = 3):
    """
    Entrena un SVM sobre els histogrames BoW i fa prediccions sobre el test.
    Utilitza GridSearchCV per buscar bons hiperparàmetres (C i gamma).
    """

    # 1. Convertim DataFrame/Series a arrays numpy
    # X: cada fila és un histograma BoW (característiques), y: etiqueta (nom del plat)
    
    X_train = X_train_df.values #.astype(float) #per evitar problemes de tipus
    X_test = X_test_df.values #.astype(float)
    y_train = Y_train_ser.values
    

    # 4. Definim el model SVM bàsic
    svm = SVC(C=C, class_weight='balanced', gamma=gamma, kernel=kernel, degree=degree)
    # el rbf és el que funciona millor per a reconeixement d'imatges
    # la C és el que varia, C gran penalitza molt els errors i pot sobreajustar, C petita accepta més errors i pot generalitzar millor, però també ajustar poc
    #class_weight='balanced' el que fa és donar més pes a les classes minoritàries per evitar biaixos i menys pes a les majoritàries
    #fem servir ovr (one vs rest) per a classificació multiclasse ja que és la més ràpida  
    
    # 6. Entrenem el model sobre el train
    svm.fit(X_train, y_train)
     

    # 7. Fem les prediccions sobre el conjunt de test amb el millor model trobat
    y_pred = svm.predict(X_test)

    return y_pred


def avalua_prediccions_svm(Y_test_ser, y_pred):
    """
    Calcula i mostra:
      - l'accuracy global
      - la matriu de confusió (files = etiquetes reals, columnes = prediccions)
    Retorna (accuracy, matriu_confusio).
    """
    # Convertim a arrays de Numpy  si Y_test és una sèrie de pandas, sinó la deixa igual
    y_test = Y_test_ser.values if hasattr(Y_test_ser, "values") else Y_test_ser 

    # 1. Accuracy global
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-score:", f1)

    # 2. Matriu de confusió
    labels_ordenats = sorted(list(set(y_test)))  #per mantenir sempre el mateix ordre a la matriu de confusió
    cm = confusion_matrix(y_test, y_pred, labels=labels_ordenats)

    cm_df = pd.DataFrame(cm, index=labels_ordenats, columns=labels_ordenats)#ho passem a dataframe per poder guardar-ho millor
    
 
    return acc, cm_df, prec, rec, f1

#llista dels fitxers que donen error
