from train_test_split import train_test_split_dataset 
from bow import bow_dataset_cv, bow_dataset_sklearn, bow_dataset2
from histogrames import calcula_histograma_train
from calcula_prediccio import calcula_prediccio, avalua_prediccions

test_size=0.2
MAX_DESC_PER_IMAGE = 100
K = 60

x_train_dataset, x_test_dataset, y_train, y_test = train_test_split_dataset(test_size, max_desc=MAX_DESC_PER_IMAGE)
#això són series de pandas, ja que només tenim 1 columna, té 1 sola columna, però cada element té un index
#cada descriptor és una matriu de (N,128), on N és el nombre de descriptors extrets de la imatge (com a màxim MAX_DESC_PER_IMAGE)

#print(x_train_dataset.values)  #això és una llista de matrius numpy

bow_centers = bow_dataset_sklearn(K,x_train_dataset)
#bow_centers = bow_dataset2(x_train_dataset)
x_train = calcula_histograma_train(x_train_dataset,bow_centers) #això ens un panda dataframe on cada fila és l'histograma d'una imatge i cada columna una paraula visual
x_test = calcula_histograma_train(x_test_dataset,bow_centers)#això és el mateix però al test
y_pred = calcula_prediccio(x_train, y_train, x_test) #prediccions sobre el test

avalua_prediccions(y_test, y_pred) #avaluem les prediccions


#print(bow_centers)