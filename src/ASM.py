import cv2
import os
import dlib
import numpy as np



class ASMFaceLandmarker:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена по указанному пути: {model_path}")
        # детектор лиц
        self.detector = dlib.get_frontal_face_detector()
        # загрузка модели для поиска ориентиров
        self.predictor = dlib.shape_predictor(model_path)
        self.all_landmarks = []
        self.template_points = []


    def get_faces(self, img):
        if img is None:
            raise ValueError("Передано пустое изображение!")
    
        # Если изображение имеет альфа-канал, убираем его
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Убедимся, что тип — uint8 и массив непрерывен
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        gray = np.ascontiguousarray(gray)
    
        faces = self.detector(gray)
        if len(faces) == 0:
            faces = []
        
        return gray, faces
    

    def detect_landmarks(self, img):

        gray, faces = self.get_faces(img)

        shape = self.predictor(gray, faces[0])
        landmarks = np.array([(p.x, p.y) for p in shape.parts()])
        self.all_landmarks.append(landmarks)
        return landmarks

    def draw_landmarks(self, image, landmarks):
        """Накладывает обнаруженные ориентиры на копию изображения."""
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        return image
    
    def save_template_points(self):
        all_landmarks = np.array(self.all_landmarks)  # размер (N, 68, 2)
        # Выберем, например, индексы для левого глаза (36), правого глаза (45) и носа (33)
        left_eye_mean = np.mean(all_landmarks[:, 36, :], axis=0)
        right_eye_mean = np.mean(all_landmarks[:, 45, :], axis=0)
        nose_mean = np.mean(all_landmarks[:, 33, :], axis=0)
        points = [left_eye_mean.tolist(), right_eye_mean.tolist(), nose_mean.tolist()]
        self.template_points = points
    
    
def process_image(image_path, landmarker):
    """Обрабатывает одно изображение – находит лицо, рисует ориентиры и сохраняет результат."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Изображение не загружено")
    

    landmarks = landmarker.detect_landmaks(image)
    if landmarks is None:
        print("Лицо не обнаружено на изображении:", image_path)
        return
    
    #result_image = landmarker.draw_landmarks(image.copy(), landmarks)

    return landmarks.flatten()

