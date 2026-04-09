from ASM import ASMFaceLandmarker
import PCA_Face, KNN_Face
import os
import VideoFrame


MODEL_PATH = os.path.join('src', 'models', 'ASM.dat')
INPUT_DIR = os.path.join('src', 'img')
OUTPUT_DIR = os.path.join('src', 'img', 'results')

def Main():
    # Создание каталога для результатов, если он отсутствует
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Инициализация объекта для работы с ASM
    landmarker = ASMFaceLandmarker(MODEL_PATH)

    pca, reduced_features = PCA_Face.train_PCA(landmarker, INPUT_DIR)
    landmarker.save_template_points()

    first_labels = KNN_Face.Labels()
    
    first_labels.load_embeddings_labels("Fattahov","src/img/Fattahov", landmarker, pca)
    first_labels.load_embeddings_labels("Galkin","src/img/Galkin", landmarker, pca)
    first_labels.convert()

    knn = KNN_Face.KNN(first_labels.embeddings, first_labels.labels)

    #FaceCompare.compare_new_face('Isaev.jpg', landmarker, pca, reduced_features, 0.3)
    VideoFrame.video(knn, landmarker, pca, reduced_features, 0.5)

    #face_compare.compare_faces('src\img\Galkin2.jpg', 'src\img\Isaev.jpg', landmarker, pca)


if __name__ == "__main__":
    print(f'Запуск главного модуля..........')
    try: Main()
    except KeyboardInterrupt as e: print(f'Работа остановлена вручную: {e}')
    except Exception as e: print(f'Ошибка вида: {e}')
    finally: print(f'Прекращаю работу.................')

