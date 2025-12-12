#from train_test_split import train_test_split_dataset 
from bow import bow_dataset_cv, bow_dataset_sklearn, bow_dataset_sklearn_fast
from histogrames import calcula_histograma_train
from calcula_prediccio import calcula_prediccio, avalua_prediccions
from svm import calcula_prediccio_svm, avalua_prediccions_svm
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from train_test_split_color import train_test_split_dataset
from augmentation import augment_training_set
import shutil
from collections import Counter

step = 10
kp_size = 2.0
test_size = 0.3
gamma = "scale"
kernel = 'rbf'
degree = 3
#degree 3 ha donat bé 
k = 500
#c=10.0
augment = 0
cs = [0.01, 0.1, 1.0, 10.0, 100.0]

RESULTS_FILE = (
    f"proves_pastis_fregit.txt")


augment = 0
seed = 18
shutil.rmtree("Pickles Per Imatge en Color")

for c in cs:
    with open(RESULTS_FILE, "a") as f:
                
                print(f"Inici prova: c = {c}")
                """
                f.write("\n=============================================\n")
                
                f.write(f"Prova Desbalanceig 4 {augment} \n")
                        
                f.write("=============================================\n")
                
                """
                f.write("C = {c}")

                #print(f"Inici prova: C={c}, K={k}")
                                
                                # 1. Split
                x_train_dataset, x_test_dataset, y_train, y_test, id_train, id_test, index_map = train_test_split_dataset(
                test_size, step, kp_size)

                x_train_dataset, y_train, id_train = augment_training_set(x_train_dataset, y_train, id_train, index_map, step, kp_size, num_aug=augment,seed=seed)

                # Comptar nombre de fotos per categoria
                counter = Counter(y_train)

                print("Nombre de fotos per categoria després de l'augmentació:")
                for label, count in counter.items():
                    print(f"{label}: {count}")               
                        
                                # 2. Cluster centers
                bow_centers, kmeans_model = bow_dataset_sklearn_fast(k, x_train_dataset)
                                

                                
                print("Calculant histogrames")
                                # 3. Histogrames
                x_train = calcula_histograma_train(x_train_dataset, kmeans_model, k)
                x_test = calcula_histograma_train(x_test_dataset, kmeans_model,k)
                        
                                # 4. Prediccions
                y_pred = calcula_prediccio_svm(
                        x_train, y_train, x_test, C=c, gamma=gamma, kernel=kernel, degree=degree)
                acc, cm_df, prec, rec, f1 = avalua_prediccions_svm(y_test, y_pred)
                                
                               
                # 5. Guardem accuracy
                f.write(f"Accuracy:  {acc:.4f}\n")
                f.write(f"Precision: {prec:.4f}\n")
                f.write(f"Recall:    {rec:.4f}\n")
                f.write(f"F1-score:  {f1:.4f}\n\n")

                # 6. Guardem la matriu de confusió (text)
                f.write("Matriu de confusió:\n")
                f.write(cm_df.to_string())
                f.write("\n\n")

                      
                                
                # 7. Guardem la CM com a imatge
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"Confusion Matrix rfb C = {c}")
                fig_name = f"Confusion Matrix rfb C = {c}.png"
                plt.savefig(fig_name, dpi=300, bbox_inches='tight')
                plt.close()
                
                                