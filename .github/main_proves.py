from train_test_split_color import train_test_split_dataset 
from bow import bow_dataset_cv, bow_dataset_sklearn, bow_dataset_sklearn_fast
from histogrames import calcula_histograma_train
from calcula_prediccio import calcula_prediccio, avalua_prediccions
from svm import calcula_prediccio_svm, avalua_prediccions_svm
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutill

step = 10
kp_size = 2.0
test_size = 0.3
gamma = "scale"
kernel = 'rbf'
degree = 2
k = 500
c=100.0


RESULTS_FILE = (
    f"VARIA_GAMMA_DEGREE_fixe_K{k}_c{c}_step{step}_kp_size{kp_size}_test{test_size}_"
    f"kernel{kernel}.txt"
)

shutil.rmtree("Pickles Per Imatge en Color")


with open(RESULTS_FILE, "a") as f:
                """                
                print(f"Inici prova: C={c}, K={k}")
                
                f.write("\n=============================================\n")
                
                f.write(f"Prova: C={c}, K={k}, step={step}, kp_size={kp_size}, test_size={test_size}, "
                        f"gamma={gamma}, kernel={kernel}, degree={degree}\n")
                f.write("=============================================\n\n")
                """
                # 1. Split
                x_train_dataset, x_test_dataset, y_train, y_test, id_train, id_test = train_test_split_dataset(
                    test_size, step, kp_size)
                
               
                # 2. Cluster centers
                bow_centers, kmeans_model = bow_dataset_sklearn_fast(k, x_train_dataset)
                primera = False

                    
                print("Calculant histogrames")
                # 3. Histogrames
                x_train = calcula_histograma_train(x_train_dataset, kmeans_model, k)
                x_test = calcula_histograma_train(x_test_dataset, kmeans_model,k)
            
                # 4. Prediccions
                y_pred = calcula_prediccio_svm(
                    x_train, y_train, x_test, C=c, gamma=gamma, kernel=kernel, degree=degree)
                acc, cm_df, prec, rec, f1 = avalua_prediccions_svm(y_test, y_pred)
                """
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
                plt.title(f"Confusion Matrix C={c}, K={k}, step={step}, kp_size={kp_size}, test_size={test_size}, "
                        f"gamma={gamma}, kernel={kernel}, degree={degree} ")
                fig_name = f"confusion_matrix_C{c}_K{k}_gamma_{gamma}_kernel{kernel}.png"
                plt.savefig(fig_name, dpi=300, bbox_inches='tight')
                plt.close()
                """
                
                

        
         

