from train_test_split_color_histogram import train_test_split_dataset
from bow import  bow_dataset_sklearn_fast
from histogrames import calcula_histograma_train
from dues_opcions import calcula_prediccio_dues_opcions, avalua_prediccions_dues_opcions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

k = 2000
step = 12 
block_size = 16
test_size = 0.3
gamma = 'scale'
kernel = 'hist_intersection'
c = 0.5

x_train_dataset, x_val_dataset, x_test_dataset, y_train, y_val, y_test, id_train, id_val, id_test = train_test_split_dataset(
                        test_size, step, block_size)
                    
                
# 2.Cluster centers
bow_centers, kmeans_model = bow_dataset_sklearn_fast(k, x_train_dataset)
                    

                        
print("Calculant histogrames")
# 3.Histogrames
x_train = calcula_histograma_train(x_train_dataset, kmeans_model, k)
x_test = calcula_histograma_train(x_test_dataset, kmeans_model,k)
x_val = calcula_histograma_train(x_val_dataset, kmeans_model,k)


#DADES DE TEST
# 4.Prediccions
y_pred = calcula_prediccio_dues_opcions(
                        x_train, y_train, x_test, C=c, gamma=gamma, kernel='hist_intersection')

acc = avalua_prediccions_dues_opcions(y_test, y_pred)



                
                    

