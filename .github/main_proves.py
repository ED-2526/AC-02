from train_test_split import train_test_split_dataset 
from bow import bow_dataset_cv, bow_dataset_sklearn, bow_dataset_sklearn_fast
from histogrames import calcula_histograma_train
from calcula_prediccio import calcula_prediccio, avalua_prediccions
from svm import calcula_prediccio_svm, avalua_prediccions_svm




C = 1.0
K = 500
step = 30
kp_size = 5.0
test_size = 0.2
C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
K_list = [100, 300, 500, 1000, 2000]



RESULTS_FILE = f"proves_{step}_pixels.txt"
with open(RESULTS_FILE, "w") as f:

    for c in C_list:
        for k in K_list:


            print(f"Inici prova: C={c}, K={k}")

            f.write("\n=============================================\n")
            f.write(f"   Prova: C = {c},  K = {k}\n")
            f.write("=============================================\n\n")

            # 1. Split
            x_train_dataset, x_test_dataset, y_train, y_test, id_train, id_test = train_test_split_dataset(test_size,step)

            # 2. Cluster centers
            bow_centers = bow_dataset_sklearn_fast(k, x_train_dataset)

            # 3. Histogrames
            x_train = calcula_histograma_train(x_train_dataset, bow_centers)
            x_test = calcula_histograma_train(x_test_dataset, bow_centers)

            # 4. Prediccions
            y_pred = calcula_prediccio_svm(x_train, y_train, x_test, C=c)
            acc, cm_df = avalua_prediccions_svm(y_test, y_pred)

            # 5. Guardem accuracy
            f.write(f"Accuracy: {acc:.4f}\n\n")

            # 6. Guardem la matriu de confusió en format txt
            f.write("Matriu de confusió:\n")
            f.write(cm_df.to_string())  
            f.write("\n\n")

            print(f"Prova acabada: C={c}, K={k}  -->  Accuracy: {acc:.4f}")

            # 7. Guardem imatges mal classificades
            f.write("Males classificacions:\n")

            for i, (real, pred) in enumerate(zip(y_test, y_pred)):
                if real != pred:
                    f.write(f"  ID: {id_test.iloc[i]}  --> Real: {real},  Pred: {pred}\n")

            f.write("\n")





