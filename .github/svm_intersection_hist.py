from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

# ==========================================================
# DEFINICIÓ DEL KERNEL D'INTERSECCIÓ D'HISTOGRAMES
# ==========================================================
def histogram_intersection_kernel(X, Y):
    """
    Calcula la intersecció d'histogrames: K(x, y) = sum(min(x_i, y_i))
    X: matriu (n_samples_1, n_features)
    Y: matriu (n_samples_2, n_features)
    Retorna: matriu Kernel (n_samples_1, n_samples_2)
    """
    K = np.zeros((X.shape[0], Y.shape[0])) #matriu buida on apuntarem similitud entre cada imatge de train i cada imatge de test
    for i, x in enumerate(X): #agafa una imatge del primer grup
        # min entre un vector x i tota la matriu Y
        K[i, :] = np.sum(np.minimum(x, Y), axis=1)
    return K


def calcula_prediccio_svm_intersection_hist(X_train_df, Y_train_ser, X_test_df, C, gamma, kernel, degree=3):
    """
    Entrena un SVM sobre els histogrames BoW.
    Accepta kernel='hist_intersection' per utilitzar el custom kernel.
    """

    # converteix les taules de Pandas a arrays numpy i  a float perque l'SVM només entèn aquest tipus
    X_train = X_train_df.values.astype(float)
    X_test = X_test_df.values.astype(float)
    y_train = Y_train_ser.values
    
    # Triem el kernel
    if kernel == 'hist_intersection':
        print("Utilitzant Histogram Intersection Kernel...")
        svm_kernel = histogram_intersection_kernel #fem funció anterior que hem creat
    else:
        svm_kernel = kernel

    svm = SVC(C=C, 
              class_weight='balanced', 
              kernel=svm_kernel, 
              gamma=gamma, 
              degree=degree)
    
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    return y_pred


def avalua_prediccions_svm(Y_test_ser, y_pred):
    """
    Calcula mètriques i matriu de confusió.
    """
    y_test = Y_test_ser.values if hasattr(Y_test_ser, "values") else Y_test_ser  # si les dades tenen atribut values, l'agafem. Així evitem errors de compatiblitat amb Pandas o Numpy

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    
    labels_ordenats = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, y_pred, labels=labels_ordenats)
    cm_df = pd.DataFrame(cm, index=labels_ordenats, columns=labels_ordenats)
 
    return acc, cm_df, prec, rec, f1