from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calcula_prediccio_xgboost(X_train_df, Y_train_ser, X_test_df, Y_test_ser, n_estimators, max_depth, learning_rate):
    """
    Entrena XGBoost i genera una gràfica d'evolució de l'ACCURACY (Train vs Test).
    S'han afegit paràmetres de regularització per combatre l'overfitting.
    """
    
    X_train = X_train_df.values
    X_test = X_test_df.values

    le = LabelEncoder() # XGBoost no enten text i codifica els labels en números
    y_train_encoded = le.fit_transform(Y_train_ser)
    y_test_encoded = le.transform(Y_test_ser) 
    
    model = XGBClassifier(
        n_estimators=n_estimators, #quants arbres de decisió creem
        max_depth=max_depth, # la màxima profunditat d'aquests arbres
        learning_rate=learning_rate,
        objective='multi:softmax', #indiquem que el problema és multiclasse
        random_state=42, #obtenim sempre els mateixos resultats
        eval_metric=['merror', 'mlogloss'], 
        n_jobs=-1 
    )

    # 4. Entrenar
    print(f"Entrenant XGBoost (Accuracy Plot) amb regularització...")
    
    eval_set = [(X_train, y_train_encoded), (X_test, y_test_encoded)] #li passem dades de test i train per veure si hi ha overfitting    
    model.fit(
        X_train, 
        y_train_encoded, 
        eval_set=eval_set, 
        verbose=False 
    )

    results = model.evals_result()
    epochs = len(results['validation_0']['merror'])
    x_axis = range(0, epochs)

    # recuperem l'error i el passem a accuracy
    train_error = results['validation_0']['merror']
    test_error = results['validation_1']['merror']
    train_acc = [(1 - x) * 100 for x in train_error]
    test_acc = [(1 - x) * 100 for x in test_error]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_axis, train_acc, label='Train Accuracy')
    ax.plot(x_axis, test_acc, label='Test Accuracy')
    
    ax.legend()
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Nombre d\'arbres (n_estimators)')
    plt.title('XGBoost Learning Curve: Accuracy Evolution (Regularized)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("xgboost_accuracy_curve_regularized.png")
    plt.close()
    print("Gràfica guardada com 'xgboost_accuracy_curve_regularized.png'")

    y_pred_encoded = model.predict(X_test) #predim les dades de test i les descodifiquem per a entendre els resultats, ja que XGBoost no "treballa" amb text.
    y_pred = le.inverse_transform(y_pred_encoded)

    return y_pred

def avalua_prediccions_xgboost(Y_test_ser, y_pred):

    y_test = Y_test_ser.values if hasattr(Y_test_ser, "values") else Y_test_ser # si les dades tenen atribut values, l'agafem. Així evitem errors de compatiblitat amb Pandas o Numpy

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"XGBoost Resultats -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

    labels_ordenats = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, y_pred, labels=labels_ordenats)
    cm_df = pd.DataFrame(cm, index=labels_ordenats, columns=labels_ordenats)

    return acc, cm_df, prec, rec, f1