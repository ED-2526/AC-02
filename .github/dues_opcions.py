from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns



def histogram_intersection_kernel(X, Y):
    """
    Calcula la intersecció d'histogrames: K(x, y) = sum(min(x_i, y_i))
    X: matriu (n_samples_1, n_features)
    Y: matriu (n_samples_2, n_features)
    Retorna: matriu Kernel (n_samples_1, n_samples_2)
    """
    # Si les matrius són molt grans, fer-ho amb broadcasting pot consumir molta RAM.
    # Aquesta versió és segura i raonablement ràpida:
    
    # Opció B (Iterativa, més segura per a la memòria):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        # min entre un vector x i tota la matriu Y
        K[i, :] = np.sum(np.minimum(x, Y), axis=1)
    return K




def calcula_prediccio_dues_opcions(X_train_df, Y_train_ser, X_test_df, C, gamma='scale', kernel='rbf', degree=3):
    """
    Entrena un SVM i retorna prediccions. 
    Si la diferència entre les dues millors classes és < 10%, retorna ambdues.
    """

    #aqui fem el mateix que el classificaodor SVM normal 
    X_train = X_train_df.values 
    X_test = X_test_df.values 
    y_train = Y_train_ser.values

    
    svm = SVC(C=C, 
              class_weight='balanced', 
              gamma=gamma, 
              kernel=histogram_intersection_kernel if kernel=='hist_intersection' else kernel, 
              degree=degree, 
              probability=True) 
    
    
    svm.fit(X_train, y_train)

    #obtenim les probabilitats per a cada classe
    y_proba = svm.predict_proba(X_test)
    
    #obtenim els noms de les classes(per saber a què correspon cada índex)
    classes = svm.classes_
    
    # ---------------------------------------------------------
    # NOVA LÒGICA
    # ---------------------------------------------------------
    y_pred_custom = [] #la llista final

    for probs in y_proba:
        #np.argsort retorna els índexs ordenats de menor a major probabilitat
        #pillem els dos últims 
        indices_ordenats = np.argsort(probs)
        idx_best = indices_ordenats[-1]      #el millor
        idx_second = indices_ordenats[-2]    #el segon millor

        prob_best = probs[idx_best] #la millor probabilitat
        prob_second = probs[idx_second] #la segona millor probabilitat

        gap = prob_best - prob_second #trobem la diferencia

        
        if gap < 0.3: #aqui especifiquem la diferencia de probabilitats entre el primer i el segon
            prediccio = [classes[idx_best], classes[idx_second]]#guardem una llista amb les dues etiquetes
        else:
            prediccio = classes[idx_best] #guarde la millor etiqueta com a string
        
        y_pred_custom.append(prediccio)

    
    return y_pred_custom




def avalua_prediccions_dues_opcions(y_true, y_pred_custom, labels=None):
    """
    Calcula l'accuracy 'Best-of-2', genera la matriu de confusió 
    i la guarda com a imatge (PNG) seguint el teu format específic.
    """
    
    #assegurem que y_true sigui iterable
    if hasattr(y_true, 'values'):
        y_true = y_true.values
        
    encerts = 0 #comptador d'encerts
    total = len(y_true) #nombre total de mostres
    dubtes = 0 #comptador de dubtes
    
    #llista per poder fer la matriu de confusió
    y_pred_per_matriu = []

    
    for etiqueta_real, prediccio in zip(y_true, y_pred_custom):
            
            #si tenim una llista i per tant dubte
            if isinstance(prediccio, list):
                dubtes += 1                
                #si la realitat està dins les opcions, ho comptem com a encert perfecte
                if etiqueta_real in prediccio:
                    encerts += 1
                    y_pred_per_matriu.append(etiqueta_real) 
                else:
                    #si falla les dues, agafem la primera opció com a error
                    y_pred_per_matriu.append(prediccio[0])

            #si tenim string i per tant no hi ha dubte
            else:
                y_pred_per_matriu.append(prediccio)
                if etiqueta_real == prediccio:
                    encerts += 1
                    
    #calculem l'accuracy
    accuracy = encerts / total if total > 0 else 0

    print(f"Dubtes (gap): {dubtes} de {total}")
    print(f"Total mostres: {total}")
    print(f"Encerts (relaxat): {encerts}")
    print(f"Accuracy: {accuracy:.4f}")
        
   
    """ SI VOLEM GUARDAR LA MATRIU DE CONFUSIÓ COM A IMATGE
    #generem la Matriu de Confusió
    cm = confusion_matrix(y_true, y_pred_per_matriu, labels=labels)
        
    # creem el DataFrame perquè el heatmap tingui els noms 
    if labels is not None:
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    else:
        # Si no passes labels, fem servir els únics trobats (ordenats alfabèticament)
        unique_labels = sorted(list(set(y_true) | set(y_pred_per_matriu)))
        cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)

    #el teu bloc de codi per visualitzar i guardar
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
        
    # He canviat el títol a SVM GAP per coherència, però pots posar el que vulguis
    plt.title(f"Matriu confusió dubtes GAP = 30 Acc: {accuracy:.2f})")
        
    fig_name = f"Matriu_confusio_SVM_GAP.png"
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.close() # Tanquem la figura perquè no es superposi si fas un bucle
        
    print(f"Gràfic guardat com: {fig_name}")
    """ 
    return accuracy