Aquesta és la carpeta del grup 2 (Laia Espluga, Abel Guàrdia i Gerard Folch) del projecte d'Indian Food. Aquesta carpeta conté tots
els fitxers i datasets que hem desenvolupat durant el projecte.

Els fitxers que trobem son:

train_test_split_grisos.py: extracció de descriptors SIFT i divisió de les dades.
train_test_split_mean.py: extracció de descriptors en color (mètode de mitjana i desviació) i divisió de les dades.
train_test_split_histogram.py: extracció de descriptors en color (mètode d'histogrames de color per blocs) i divisió de les dades.
bow.py: creació del vocabulari visual amb KMeans, a partir del conjunt d'entrenament.
histogrames.py: creació dels histogrames a partir del Bag of Words creat.
calcula_prediccio_KNN.py: classificació i posterior avaluació amb KNN.
svm_rbf_kernel.py: classificació i posterior avaluació amb SVM i kernel rbf.
svm_intersection_hist.py: classificació i posterior avaluació amb SVM i kernel Histogram Intersection.
xgboost_model.py: classificació i posterior avaluació amb el model XGBoost.
double_step_svm.py: classificació i posterior avaluació amb el classificador Double Step (classificador específic per les categories en conflicte).
dues_opcions.py: classificació i posterior avaluació amb el classificador Dues Opcions (valorar les dues respostes més probables que retorna l'SVM).

MAINS:

MAIN_SVM.py: main de proves amb classificador SVM amb kernel d'Histogram Intersection.
MAIN_dues_opcions.py: main de proves amb classificador SVM de Dues Opcions.
MAIN_doble_classificador.py: main de proves amb classificador SVM de Double Step (un segon classificador SVM "expert" en els conflictes).

CARPETES:

FITXERS AUGMENTATION: carpeta amb els fitxers necessaris per a dur a terme la Data Augmentation.
PROVES: proves que hem fet a la Versió 2, 3 i 4.