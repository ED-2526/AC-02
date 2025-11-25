import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def train_test_split_dataset(test_size):
    base_path = os.getcwd()
    print("CWD:", base_path)

    path = os.path.join(base_path, "Indian Food Images")
    print("PATH:", path)

    # llista totes les carpetes del directori, que són les classes del dataset
    labels = os.listdir(path)

    # crea un dataframe buit amb dues columnes, path i label
    df = pd.DataFrame(columns=['img_path', 'label'])

    # recorrem les carpetes (cada carpeta = una classe)
    for label in labels:
        img_dir_path = os.path.join(path, label)  # path de cada carpeta
        for img in os.listdir(img_dir_path):      # recorrem les imatges de cada carpeta
            img_path = os.path.join(img_dir_path, img)  # path de cada imatge
            df.loc[df.shape[0]] = [img_path, label]     # afegim fila al dataframe

    print(df.head())  # mostra les primeres files del dataframe

    # split estratificat: mateix percentatge per classe en train i test
    x_train, x_test, y_train, y_test = train_test_split(
        df['img_path'],
        df['label'],
        test_size=test_size,
        random_state=42,
        stratify=df['label']
    )

    return x_train,x_test,y_train,y_test

if __name__ == "__main__":
    # Proves només si s'executa aquest fitxer directament
    x_train, x_test, y_train, y_test = train_test_split_dataset(test_size=0.2)

    print("Train:", len(x_train))
    print("Test:", len(x_test))
    print("Classes al TRAIN:", sorted(list(set(y_train))))
    print("Classes al TEST:", sorted(list(set(y_test))))
    
    print("Distribució Train:")
    print(y_train.value_counts())
    print("Distribució Test:")
    print(y_test.value_counts())