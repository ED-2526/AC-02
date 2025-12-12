import cv2
import numpy as np
import random
from numpy.lib.stride_tricks import as_strided

def get_sift_descriptors_from_image(img, step, block_size):
    if img is None:
        return None

    h, w, c = img.shape
    bs = int(block_size)
    st = int(step)

    ny = (h - bs) // st + 1
    nx = (w - bs) // st + 1

    stride_y, stride_x, stride_c = img.strides

    blocks = as_strided(
        img,
        shape=(ny, nx, bs, bs, c),
        strides=(st * stride_y, st * stride_x, stride_y, stride_x, stride_c),
        writeable=False
    )

    means = blocks.mean(axis=(2, 3))
    stds  = blocks.std(axis=(2, 3))

    descriptors = np.concatenate([means, stds], axis=-1)
    return descriptors.reshape(-1, 6)


def augment_training_set(x_train, y_train, id_train, index_map, step, kp_size, num_aug=1,seed=42):
    new_descriptors = []
    new_labels = []
    new_ids = []

    next_id = max(index_map.keys()) + 1
    seed_global = seed 

    def augment_image(img, image_index, augment_index):
        # Seed específica per augment
        seed_local = seed_global + image_index + augment_index
        random.seed(seed_local)
        np.random.seed(seed_local)

        aug = img.copy()

        # Flip horitzontal
        if random.random() < 0.5:
            aug = cv2.flip(aug, 1)

        # Rotació [-20°, +20°]
        angle = random.uniform(-20, 20)
        
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # Canvi d'il·luminació (gamma)
        gamma = random.uniform(0.7, 1.3)
        look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        aug = cv2.LUT(aug, look_up_table)

        # Blur suau
        if random.random() < 0.3:
            aug = cv2.GaussianBlur(aug, (5, 5), 0)

        return aug

    # ===========================
    # Loop sobre les imatges del train
    # ===========================
    for i, (label, img_id) in enumerate(zip(y_train, id_train)):
        img_path = str(index_map[img_id])
        img = cv2.imread(img_path)
        if label == "DOLCOS_FREGITS":
            continue
        if img is None:
            print("ERROR augmentant:", img_path)
            continue

        for aug_idx in range(num_aug):
            img_aug = augment_image(img, image_index=img_id, augment_index=aug_idx)
            descriptors_aug = get_sift_descriptors_from_image(img_aug, step, kp_size)
            if descriptors_aug is None:
                continue

            new_descriptors.append(descriptors_aug)
            new_labels.append(label)
            new_ids.append(next_id)
            next_id += 1

    # ===========================
    # Combinar originals + augmentats
    # ===========================
    X_train_final = list(x_train) + new_descriptors
    Y_train_final = list(y_train) + new_labels
    ID_train_final = list(id_train) + new_ids

    return X_train_final, Y_train_final, ID_train_final
