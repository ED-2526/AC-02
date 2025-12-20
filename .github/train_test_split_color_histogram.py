import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import pickle
import csv
import cv2
import numpy as np
from skimage.feature import local_binary_pattern 
from numpy.lib.stride_tricks import as_strided

def color_hist_block(img_path, step=8, block_size=16, bins=8):
   
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w, c = img.shape #obtenim les dimensions
    bs, st = block_size, step

    #Nombre de blocs
    ny = (h - bs) // st + 1 #nombre de blocs en vertical 
    nx = (w - bs) // st + 1 #nombre de blocs en horitzontal 

    stride_y, stride_x, stride_c = img.strides#això és que numpy guarda la matriu en una fila i això et diu quan has de saltar per anar a la següent fila, a la seguent columna o al següent color

    blocks = as_strided(
        img,
        shape=(ny, nx, bs, bs, c),
        strides=(st*stride_y, st*stride_x, stride_y, stride_x, stride_c),
        writeable=False
    )
    #a shape tenim:
    # ny = indicador vertial (1 fins ny-1), nx = indicador horitzonal (1 fins nx-1), bs són files i columnes dins del bloc i la c el nombre de components 
    #el que fem és reinterpretar la memòria de la imtage, però no modifiquem la original
    #ara en comptes de guardar la imatge per ordre el que fem és guardem els blocs en ordre de manera consecutiva

    
    q = (blocks.astype(np.int16) * bins) // 256 #passem de tenir 256 valors a tenir-ne només 8 per fer-ho més manejable
    q = np.clip(q, 0, bins - 1)

    descriptors = []
    for ch in range(c): #anem per cada color i per cada color anem fent l'histograma, és a dir anirem comptant quants pixels estan entre 0-31, 32-63, etc per cafa color
        hist = np.eye(bins)[q[..., ch]].sum(axis=(2, 3)) #anem per cada canal i cada pixel el posem com un vector de un 1 i la resta 0, després fem la suma per tenir l'histograma
        descriptors.append(hist) #afegim l'histograma d'aquest canal a la llista de descriptors

    descriptors = np.concatenate(descriptors, axis=-1) #ajuntem els histogrames de cada canal de color

    return descriptors.reshape(-1, c * bins).astype(np.float32) #cada fila és un bloc i cada columna és un bin d'un canal de color


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



# ============================================================
# 3. Funció principal per processar dataset
# ============================================================
def train_test_split_dataset(test_size,step, kp_size):
    base_path = os.getcwd() #directori actual
    images_path = os.path.join(base_path, "DATASETS\\Indian Food Generalitzat Balancejat Resize") #carpeta amb les imatges
    pickle_root = os.path.join(base_path, "Pickles Per Imatge en Color 2") #carpeta on guardarem els pickles

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
                    descriptors = color_hist_block(img_path, step, kp_size) #aqui extraiem els descriptors SIFT
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

    x_train, x_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        df["descriptors"],
        df["label"],
        df["image_index"],
        test_size=0.3,
        random_state=42,
        stratify=df["label"]
    )
    
    # split temp en validation i test
    
    x_val, x_test, y_val, y_test, id_val, id_test = train_test_split(
        x_temp, y_temp, id_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )
    
    return x_train, x_val, x_test, y_train, y_val, y_test, id_train, id_val, id_test
    #ara el que volem endivinar és el label, que és el plat general (pa, postres, dolços, etc), si volem endivinar el plat concret, hauríem de posar "plat" en comptes de "label"


