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
def get_sift_descriptors_for_image(img_path, max_desc):
    """
    Extreu descriptors SIFT DENSOS:
    - Es crea una graella regular de keypoints cada 'step' píxels.
    - Per cada punt de la graella es calcula un descriptor SIFT.
    - Si hi ha més descriptors que max_desc, se'n seleccionen aleatòriament max_desc.
    """
    sift = cv2.SIFT_create()

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape

    # Paràmetres de la graella
    step = 5        # un keypoint cada 5 píxels
    kp_size = 5.0   # "mida" del keypoint (pot ajustar-se)

    keypoints = []
    # Recorrem la imatge amb una graella regular
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Creem un keypoint a (x, y) amb la mida indicada
            keypoints.append(cv2.KeyPoint(float(x), float(y), kp_size))

    # A diferència de detectAndCompute, aquí NOMÉS fem compute()
    # perquè els keypoints els definim nosaltres (graella densa)
    keypoints, descriptors = sift.compute(img, keypoints)

    if descriptors is None or len(descriptors) == 0:
        return None

    # Limitem nombre descriptors (per no rebentar memòria)
    if descriptors.shape[0] > max_desc:
        idx = np.random.choice(descriptors.shape[0], max_desc, replace=False)
        descriptors = descriptors[idx]

    return descriptors

# ============================================================
# 2. Funció per guardar un pickle per imatge
# ============================================================
def save_image_pickle(descriptors, label, image_index, output_path):
    data = {
        "image_index": image_index,  # ara és un número
        "label": label,
        "descriptors": descriptors
    }
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

# ============================================================
# 3. Funció principal per processar dataset
# ============================================================
def train_test_split_dataset(test_size=0.2, max_desc=50):
    base_path = os.getcwd()
    images_path = os.path.join(base_path, "Indian Food Images")
    pickle_root = os.path.join(base_path, "Pickles Per Imatge")

    os.makedirs(pickle_root, exist_ok=True)

    index_map = {}     # Diccionari: image_index → path original
    df_rows = []       # Cada entrada: descriptors, label, image_index

    # Comptador d’imatges
    image_counter = 1

    # Recorrem carpetes (classes)
    labels = sorted([d for d in os.listdir(images_path)
                     if os.path.isdir(os.path.join(images_path, d))])

    print("Processant dataset i generant pickles amb index numèric...")

    for label in labels:
        class_img_folder = os.path.join(images_path, label)
        class_pickle_folder = os.path.join(pickle_root, label)
        os.makedirs(class_pickle_folder, exist_ok=True)

        for img_name in os.listdir(class_img_folder):
            img_path = os.path.join(class_img_folder, img_name)
            image_index = image_counter
            pickle_path = os.path.join(class_pickle_folder, f"{image_index}.pkl")

            # ✔️ Si el pickle JA existeix → Només el carreguem
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                descriptors = data["descriptors"]
                df_rows.append([descriptors, label, image_index])
                index_map[image_index] = img_path
                image_counter += 1
                continue

            # ✔️ Sino → Càlcul SIFT + guardat pickle
            descriptors = get_sift_descriptors_for_image(img_path, max_desc)
            if descriptors is None:
                continue

            save_image_pickle(descriptors, label, image_index, pickle_path)
            df_rows.append([descriptors, label, image_index])
            index_map[image_index] = img_path
            image_counter += 1

    # Guardem índex global en CSV
    csv_path = "image_index.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_index", "path"])  # capçalera
        for idx, path in index_map.items():
            writer.writerow([idx, path])
    print(f"Index global guardat a {csv_path}")

    # Convertim a DataFrame
    df = pd.DataFrame(df_rows, columns=["descriptors", "label", "image_index"])

    # Train / Test split (amb stratify)
    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
        df["descriptors"],
        df["label"],
        df["image_index"],
        test_size=test_size,
        random_state=42,
        stratify=df["label"]
    )

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