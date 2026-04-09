import cv2, numpy as np
import FaceCompare
from sklearn.neighbors import KNeighborsClassifier
import os

class Labels:
    
    def __init__(self):
        self.embeddings = []  # список эмбеддингов
        self.labels = []      # соответствующие метки

    def load_embeddings_labels(self, label, path, landmarker, pca):
        for filename in os.listdir(path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(path, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Не удалось прочесть изображение: {img_path}")
                    continue  # Если изображение не загрузилось, переходим к следующему
                emb = FaceCompare.get_embedding(img, landmarker, pca)
                if emb is not None:
                    self.embeddings.append(emb)
                    self.labels.append(label)

    def convert(self):
        self.embeddings = np.array(self.embeddings) # Преобразуем список эмбеддингов в numpy-массив


class KNN:
    def __init__(self, embeddings, labels):
        # Обучаем метод ближайших соседей
        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.knn.fit(embeddings, labels)

    def idenfity_people(self, img, landmarker, pca):

        #new_img = cv2.imread(img)
        new_embedding = FaceCompare.get_embedding(img, landmarker, pca)

        if new_embedding is not None:
            predicted_label = self.knn.predict(new_embedding.reshape(1, -1))
            #print("Лицо принадлежит:", predicted_label[0])
            return True, predicted_label[0]
        else:
            #print("Лицо не обнаружено")
            return False


###############################
#способ через словарь
# embeddings_dict = {
#     "Alice": embeddings_for_alice,  # np.array размером (кол-во фотографий, dim)
#     "Bob": embeddings_for_bob         # np.array размером (кол-во фотографий, dim)
# }

# # Вычисляем центроиды:
# centroids = {person: np.mean(embs, axis=0) for person, embs in embeddings_dict.items()}

# # Функция для вычисления евклидова расстояния
# def euclidean_distance(vec1, vec2):
#     return np.linalg.norm(vec1 - vec2)

# # Для нового эмбеддинга определяем ближайший центроид:
# distances = {person: euclidean_distance(new_embedding, centroid)
#              for person, centroid in centroids.items()}

# predicted_person = min(distances, key=distances.get)
# print("Лицо, вероятно, принадлежит:", predicted_person)