import cv2
import FaceCompare
 
def video(landmarker, pca, reduced_features, confidence=0.3):

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть видеопоток")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр, завершаем...")
            break
        
        # Детектируем лица
        gray, faces = landmarker.get_faces(frame)

        if faces:

            # Если лица обнаружены проверяем есть ли оно в базе и обводим его
            flag = FaceCompare.compare_new_face(gray, landmarker, pca, reduced_features, confidence)
            if flag:
                for face in faces:
                    # Рисуем прямоугольник вокруг опознанного лица
                    x1, y1 = face.left(), face.top()
                    x2, y2 = face.right(), face.bottom()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Отображаем обработанный кадр
        cv2.imshow("Real-time Face Processing", frame)
            
            # Прерываем цикл по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()