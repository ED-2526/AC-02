import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

def get_sift_descriptors_for_image(img_path, max_desc):
        
        # Crear objecte SIFT. És un objecte Python amb els seus mètodes. Per obtenir els descriptors, utilitzem un mètode d'aquesta classe.
        sift = cv2.SIFT_create()

        # Llegim la imatge i la convertim en una matriu numpy. La passem a gray scale perquè SIFT només funciona amb aquesta escala de colors.
        #Possible millora: utilitzar RGB.
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Obtenim punts d'interès (keypoints: llista d'objectes cv2.keypoints) i guardem la informació en l'array descriptors (array numpy (N,128)).
        #Si no hi ha descriptors, retorna None.
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is None or len(descriptors) == 0:
            return None

        # Comencem agafant tots els descriptors de la imatge. Si aquest nombre és major al llindar establert, agafem aleatòriament
        # aquest nombre de descriptors.
        if descriptors.shape[0] > max_desc:
            idx = np.random.choice(descriptors.shape[0], max_desc, replace=False)
            descriptors = descriptors[idx]


        # Normalització L2 de cada descritor.
        # En descriptors com SIFT, la intensitat global de la imatge pot fer que els valors del descriptor siguin més grans o més petits.
        # Fem que la longitud del vector sigui 1, fa que cap descriptor tingui “més pes” només perquè té valors més grans.

        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)  #norms és un array columna on cada element és la longitud del descriptor. El keepdims=True serveix per retornar norms com una matriu columna (1,N), i després poder-ho dividir per descriptors fila per fila sense errors (operacions entre matrius).
        norms[norms == 0] = 1.0 # Ho fem per no dividir després 0 entre 0.
        descriptors = descriptors / norms #dividim el vector entre la seva longitud

        return descriptors


def train_test_split_dataset(test_size, max_desc=50, df_file='descriptors_df.pkl'):
    base_path = os.getcwd() #aconseguim el path d'on estem
    path = os.path.join(base_path, "Indian Food Images") #li afegim el path de la carpeta amb les imatges

    #la primera vegada guardem el dataframe amb els descriptors i els labels, sino el carreguem directament
    if os.path.exists(df_file):
        print(f"Carregant DataFrame des de {df_file}...")
        df = pd.read_pickle(df_file)
    else:
        #labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))] #traiem els labels que son els noms de les carpetes amb la opció que funcioni per macOS
        labels = os.listdir(path) #traiem els labels que son els noms de les carpetes
        data = [] #llista on guardarem els descriptors i labels, serà una llista de llistes
        for label in labels: #recorrem cada carpeta
            img_dir_path = os.path.join(path, label) #creem el path de la carpeta
            for img in os.listdir(img_dir_path): #recorrem cada imatge de la carpeta
                img_path = os.path.join(img_dir_path, img) #creem el path de la imatge
                descriptors = get_sift_descriptors_for_image(img_path, max_desc) #extreiem els descriptors de la imatge
                if descriptors is not None: #en cas de tenir-ne
                    data.append([descriptors, label]) #afegim els descriptors i el label a la llista

        df = pd.DataFrame(data, columns=['descriptors', 'label']) #creem el dataframe amb els descriptors i labels

        # Guardem el DataFrame per ús futur
        df.to_pickle(df_file)
        print(f"DataFrame guardat a {df_file}.")

    #print(df.head())

    # fem split,però tenint en compte que la base de dades pot estar esbiaixada i per això podem el stratify
    x_train, x_test, y_train, y_test = train_test_split(
        df['descriptors'],
        df['label'],
        test_size=test_size,
        random_state=42,
        stratify=df['label']
    )

    return x_train, x_test, y_train, y_test

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

    