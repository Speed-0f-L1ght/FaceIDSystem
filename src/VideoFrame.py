import cv2
import FaceCompare
from datetime import datetime


 
def video(knn, landmarker, pca, reduced_features, confidence=0.4):

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть видеопоток")
        exit()

    state_change_threshold = 5
    state_counter = 0
    previous_state = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр, завершаем...")
            break
        
        # Детектируем лица
        gray, faces = landmarker.get_faces(frame)

        if faces:
            # Если лица обнаружены проверяем есть ли оно в базе и обводим его
            #is_matched = FaceCompare.compare_new_face(gray, landmarker, pca, reduced_features, confidence)
            is_matched, label = knn.idenfity_people(gray, landmarker, pca)
            current_state = "face_matched" if is_matched else "face_unknown"

            if is_matched:
                for face in faces:
                    # Рисуем прямоугольник вокруг опознанного лица
                    x1, y1 = face.left(), face.top()
                    x2, y2 = face.right(), face.bottom()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        else:
            current_state = "not_face"


        if current_state == previous_state:
            state_counter += 1
        else:
            state_counter = 0
        
        if state_counter == state_change_threshold:
            if current_state == "face_matched":
                now = datetime.now()
                print(f"Лицо найдено. {label} Время: ", now)
                #обращение к турникету (открыть)
            elif current_state == "face_unknown":
                print("Лица не совпадают с эталонными")

            elif current_state == "no_face":
                print("Лицо не обнаружено")
        previous_state = current_state


        # Отображаем обработанный кадр
        cv2.imshow("Real-time Face Processing", frame)
            
            # Прерываем цикл по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()