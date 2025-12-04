import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import pickle
import csv

# ============================================================
# 1. Funció per obtenir descriptors SIFT
# ============================================================
def get_sift_descriptors_for_image(img_path,step):
    """
    Extreu descriptors SIFT DENSOS:
    - Es crea una graella regular de keypoints cada 'step' píxels.
    - Per cada punt de la graella es calcula un descriptor SIFT.
    """
    sift = cv2.SIFT_create() #creem un descriptor SIFT

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape #agafem l'alçacada i amplada de la imatge

    #Paràmetres de la graella
    step = step        #un keypoint cada 5 píxels
    kp_size = 5.0   #quan al voltant mirem, mida del keypoint

    keypoints = [] #llista on recorrem els keypoints
    #Recorrem la imatge amb una graella regular
    for y in range(0, h, step):
        for x in range(0, w, step):
            keypoints.append(cv2.KeyPoint(float(x), float(y), kp_size))#gurdem el keypoint a la llista

    #perquè els keypoints els definim nosaltres (graella densa)
    #passem directament els keypoints al compute, la llista
    keypoints, descriptors = sift.compute(img, keypoints) #aqui nomes calculem els descriptors del punt que ja sabem

    if descriptors is None or len(descriptors) == 0:
        return None

    """
    if descriptors.shape[0] > max_desc:
        idx = np.random.choice(descriptors.shape[0], max_desc, replace=False)
        descriptors = descriptors[idx]
    """
    return descriptors #array de numpy de mida (N,128) on N és el nombre de descriptors SIFT extrets de la imatge

# ============================================================
# 2. Funció per guardar un pickle per imatge
# ============================================================
#output_path , 
#plat és el plat concret i label la classe general (pa, postre, dolços)
def save_image_pickle(descriptors, label, plat, image_index, pickle_path):
    data = {
        "image_index": image_index,  # ara és un número
        "label": label,
        "plat": plat,
        "descriptors": descriptors
    }
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f)

###CAL FER UNA FUNCIÓ QUE AGAFI ELS PIXELS QUE DEMANEM

# ============================================================
# 3. Funció principal per processar dataset
# ============================================================
def train_test_split_dataset(test_size=0.2,step=30):
    base_path = os.getcwd() #directori actual
    images_path = os.path.join(base_path, "Indian Food Generalitzat") #carpeta amb les imatges
    pickle_root = os.path.join(base_path, "Pickles Per Imatge") #carpeta on guardarem els pickles

    os.makedirs(pickle_root, exist_ok=True) #creem la carpeta si no existeix, on guardarem els pickles

    index_map = {}     #Diccionari: image_index → path original, image_index és un número únic per cada imatge i ajuntem amb el path de la imatge
    df_rows = []       #descriptors, label, image_index

    image_counter = 1 #comptador d'índexs únics per imatges

    #Ordenar les labels per tenir sempre el mateix ordre i que els splits siguin reproduïbles, podriem no fer-ho
    labels = sorted([
        d for d in os.listdir(images_path)
        if os.path.isdir(os.path.join(images_path, d))
    ])

   
    print("Processant dataset i generant pickles...")

    for label in labels: #PLATS GENERALS (PA, POSTRES, DOLÇOS, etc) 
        class_img_folder = os.path.join(images_path, label) #carpeta de la imatge actual
        class_pickle_folder = os.path.join(pickle_root, label) #carpeta on guardarem els pickles d'aquesta classe
        os.makedirs(class_pickle_folder, exist_ok=True) #crea la carpeta si no existeix

        
        for plat in os.listdir(class_img_folder):#ara recorrem cada PLAT 
            path_subfolder = os.path.join(class_img_folder, plat) #path de la subcarpeta actual de la imatge, no pel pickle

            if not os.path.isdir(path_subfolder):
                continue  #si no és carpeta, saltem

            #FOTOS dins de la subcarpeta
            for img_name in os.listdir(path_subfolder): 
                img_path = os.path.join(path_subfolder, img_name)
                image_index = image_counter
                pickle_path = os.path.join(class_pickle_folder, f"{image_index}.pkl") #guardem el path del ficher pickle

                # Si el fixter pickle ja existeix, només carreguem els descriptors
                if os.path.exists(pickle_path):
                    with open(pickle_path, "rb") as f:
                        data = pickle.load(f)
                    descriptors = data["descriptors"]
                    label = data["label"]
                    plat = data["plat"]
                    image_index = data["image_index"]
                    df_rows.append([descriptors, label, plat, image_index])
                    index_map[image_index] = img_path
                    image_counter += 1
                    continue #si ja existeix el pickle, saltem a la següent imatge
                else:
                    #Sinó: calcular SIFT i guardar pickle
                    descriptors = get_sift_descriptors_for_image(img_path,step) #aqui extraiem els descriptors SIFT
                    if descriptors is None:
                        continue

                    save_image_pickle(descriptors, label, plat,image_index, pickle_path)#el picke path el fiquem per saber on es guarda la variabñe
                    df_rows.append([descriptors, label,plat, image_index]) #afegim una fila al dataframe
                    index_map[image_index] = img_path #gaurdem el diccionari on la clau és l'índex i el valor el path de la imatge
                    image_counter += 1 #incrementem el comptador d'índexs

    if not os.path.exists("image_index.csv"): #si no existeix el fitxer el creem
        #Guardar índex en CSV
        csv_path = "image_index.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_index", "path"]) #és el diccionar on la clau és l'índex i el valor és el path(que creem abans)
            for idx, path in index_map.items():
                writer.writerow([idx, path])

        print(f"Index guardat a {csv_path}")

    # DataFrame
    df = pd.DataFrame(df_rows, columns=["descriptors", "label","plat", "image_index"])

    #Aqui fem el train test split
    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
        df["descriptors"],
        df["label"],
        df["image_index"], #això és el que enllaça amb el id_train i id_test
        test_size=test_size,
        random_state=42,
        stratify=df["label"]
    )
    #ara el que volem endivinar és el label, que és el plat general (pa, postres, dolços, etc), si volem endivinar el plat concret, hauríem de posar "plat" en comptes de "label"

    return x_train, x_test, y_train, y_test, id_train, id_test

# ============================================================
# 4. Exemple si s'executa directament
# ============================================================
if __name__ == "__main__":
    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split_dataset()

    print("\nTrain images:", len(x_train))
    print("Test images:", len(x_test))

    print("\nClasses al train:", sorted(list(set(y_train))))
    print("Exemple train:")
    print("  Image index:", id_train.iloc[0])
    print("  Label:", y_train.iloc[0])
    print("  Descriptors shape:", x_train.iloc[0].shape)