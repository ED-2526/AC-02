from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold




def calcula_prediccio(X_train, Y_train, X_test):

    param_grid = {
        #"n_neighbors": [100,200,300,400,500] #podriem fer un gràfic mostrant com millora l'accuracy
        "n_neighbors": [100,120,140,160,180,200]
    }
    #el nombre de veïns que volem provar per trobar el millor k

    kf = KFold(n_splits=30, shuffle=True, random_state=42) #creem les particions per fer cross-validation, barregem les particions


    knn = GridSearchCV( #buscarem el millor k amb cross-validation, això ja et crea un model de KNN amb el millor k
        KNeighborsClassifier(), #li diem que volem fer KNN
        param_grid, #els paràmetres que volem provar
        cv=kf,          #amb quantes particions fem cross-validation
        scoring="accuracy" #la mètrica que volem optimitzar 
    )

    knn.fit(X_train, Y_train) #entrenem el model amb els dades d'entrenament

    print("Millor k:", knn.best_params_["n_neighbors"]) #printem el millor k trobat
    
    y_pred = knn.predict(X_test) #fem les prediccions sobre les dades de test i retornem un array
    #y_pred és un array amb les prediccions per cada imatge de test(nom del plat) en ordre corresponent a X_test
    return y_pred

def avalua_prediccions(Y_test, y_pred):
    acc = accuracy_score(Y_test, y_pred) #comparem les prediccions (y_pred) amb les etiquetes reals (Y_test)
    print("Accuracy:", acc) #calculem la accuracy i la printem