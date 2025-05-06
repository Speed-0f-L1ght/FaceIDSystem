import numpy as np
from sklearn.decomposition import PCA
import os, cv2

import FaceCompare

def train_PCA(landmarker, input_dir, n_components=50):
    landmark_vectors = []
    photo_count = 0
    
    # Из указанного пути итеративно дастаются точки для каждого лица и созраняются в векторе
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(input_dir, filename)
            img = cv2.imread(path)
            landmarks = landmarker.detect_landmarks(img)
            if landmarks is not None:
                norm_landmarks = FaceCompare.normalize_landmarks(landmarks)
                feature_vector = FaceCompare.flatten_landmarks(norm_landmarks)
                landmark_vectors.append(feature_vector)
                photo_count += 1
                #filenames.append(filename)

    print(f'Колличество фотографий для обучения PCA: {photo_count}')

    if landmark_vectors is None:
        raise ValueError("Нет ключевых точек.")
    
    landmark_vectors = np.array(landmark_vectors)  # shape = (num_samples, 136)


    # Выбираем число компонент, например, 50 (это можно подобрать экспериментально)
    n_components = min(n_components, min(landmark_vectors.shape))
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(landmark_vectors)#тренировка PCA

    print("Уменьшённое представление имеет форму:", reduced_features.shape)
    return pca, reduced_features
    #embedding = pca.transform(feature_vector.reshape(1, -1))
        