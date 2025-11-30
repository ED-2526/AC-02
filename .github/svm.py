from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
import pandas as pd



def calcula_prediccio_svm(X_train_df, Y_train_ser, X_test_df):
    """
    Entrena un SVM sobre els histogrames BoW i fa prediccions sobre el test.
    Utilitza GridSearchCV per buscar bons hiperparàmetres (C i gamma).
    """

    # 1. Convertim DataFrame/Series a arrays numpy
    # X: cada fila és un histograma BoW (característiques), y: etiqueta (nom del plat)
    X_train = X_train_df.values.astype(float)
    X_test = X_test_df.values.astype(float)
    y_train = Y_train_ser.values

    # 2. Definim la graella d'hiperparàmetres per a l'SVM
    # C controla quant “castiguem” els errors (més gran = menys marge, més overfitting)
    # gamma controla la forma del kernel RBF (més gran = decisions més “locals”)
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.01, 0.001],
        "kernel": ["rbf"]
    }

    # 3. Validació creuada estratificada per mantenir el balanç de classes a cada partició
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 4. Definim el model SVM bàsic
    svm = SVC()

    # 5. GridSearchCV: prova totes les combinacions de C i gamma amb CV,
    #    i es queda amb la que dona millor accuracy.
    grid = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1  # intenta aprofitar tots els cores disponibles
    )

    # 6. Entrenem el model sobre el train
    grid.fit(X_train, y_train)

    print("Millors hiperparàmetres trobats:", grid.best_params_)
    print("Millor accuracy CV:", grid.best_score_)

    # 7. Fem les prediccions sobre el conjunt de test amb el millor model trobat
    y_pred = grid.predict(X_test)

    return y_pred


def avalua_prediccions_svm(Y_test_ser, y_pred):
    """
    Calcula i mostra:
      - l'accuracy global
      - la matriu de confusió (files = etiquetes reals, columnes = prediccions)
    Retorna (accuracy, matriu_confusio).
    """
    # Convertim a arrays per si Y_test és una sèrie de pandas
    y_test = Y_test_ser.values if hasattr(Y_test_ser, "values") else Y_test_ser

    # 1. Accuracy global
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    # 2. Matriu de confusió
    labels_ordenats = sorted(list(set(y_test)))  # per mantenir un ordre coherent
    cm = confusion_matrix(y_test, y_pred, labels=labels_ordenats)

    cm_df = pd.DataFrame(cm, index=labels_ordenats, columns=labels_ordenats)


    print("\nMatriu de confusió (files = reals, columnes = prediccions):")
    print("Ordre de les classes:")
    print(labels_ordenats)
    print("\n", cm_df)

    return acc, cm_df
