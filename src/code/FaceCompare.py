import numpy as np
import ASM
import cv2


def normalize_landmarks(landmarks):
    # Вычисляем центр по всем точкам
    center = np.mean(landmarks, axis=0)
    centered = landmarks - center
    
    # Например, масштабируем по расстоянию между глазами (точки 36 и 45 в 68-точечной модели)
    # Проверьте, что индексы соответствуют вашей модели
    left_eye = landmarks[36]
    right_eye = landmarks[45]
    eye_distance = np.linalg.norm(right_eye - left_eye)
    
    normalized = centered / eye_distance
    return normalized

def euclidean_distance(vec1, vec2):
    """Возвращает различие признаков(в виде векторов) двух лиц"""
    return np.linalg.norm(vec1 - vec2)


# Функция для "развертывания" ориентиров в одномерный вектор признаков
def flatten_landmarks(landmarks):
    return landmarks.flatten()  # превращает (68, 2) в (136,)

# Сравнение нового лица c обученной базой (сниженными признаками)
def compare_new_face(new_img_path, landmarker, pca, reduced_features,  threshold=0.5):
    img = cv2.imread(new_img_path)

    landmarks = landmarker.detect_landmaks(img)
    if landmarks is None:
        print("Лицо не обнаружено")
        return None

    norm_landmarks = normalize_landmarks(landmarks)
    feature_vector = flatten_landmarks(norm_landmarks)

    # Преобразуем новый вектор через обученную модель PCA:
    new_embedding = pca.transform(feature_vector.reshape(1, -1))[0]

    # Сравнение с базой: можно вычислить расстояние до каждого эталонного эмбеддинга
    distances = [euclidean_distance(new_embedding, ref_emb) for ref_emb in reduced_features]
    min_index = np.argmin(distances)
    min_distance = distances[min_index]

    print(f"Минимальная дистанция: {min_distance:.4f}")
    if min_distance < threshold:
        print(f"Лицо найдено")
    else:
        print("Лица не совпадают с эталонными")
    #return min_distance
    

    # Функция, которая сравнивает два лица по их признакам (feature vectors)
def compare_faces(image_path1, image_path2, landmarker, pca, threshold=0.5):
    # Загружаем изображения
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    # Извлекаем ориентиры для каждого изображения
    landmarks1 = landmarker.detect_landmaks(img1)
    landmarks2 = landmarker.detect_landmaks(img2)

    if landmarks1 is None or landmarks2 is None:
        print("Не удалось обнаружить лицо хотя бы на одном изображении.")
        return None

    # Нормализуем ориентиры (чтобы компенсировать различие по масштабу и положению)
    norm_landmarks1 = normalize_landmarks(landmarks1)
    norm_landmarks2 = normalize_landmarks(landmarks2)

    # Преобразуем в одномерные векторы признаков
    feat1 = flatten_landmarks(norm_landmarks1)
    feat2 = flatten_landmarks(norm_landmarks2)

    embedding1 = pca.pca.transform(feat1.reshape(1, -1))
    embedding2 = pca.pca.transform(feat2.reshape(1, -1))

    # Вычисляем евклидову дистанцию
    dist = euclidean_distance(embedding1, embedding2)
    print(f"Евклидова дистанция между лицами: {dist:.4f}")

    if dist < threshold:
        print("Лица принадлежат одному человеку.")
    else:
        print("Лица принадлежат разным людям.")
    
    return dist