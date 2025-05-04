# import menpo.io as mio
# import menpofit as mpfit
# from menpodetect.opencv import OpenCVDetector

# #im = mio.import_image('Galkin1.jpg', as_gray=True)
# im = mio.import_image('Galkin1.jpg')
# # Преобразование в градации серого (если требуется)
# im_gray = im.as_greyscale()

# detector = OpenCVDetector(model='haarcascade_frontalface_default.xml')
# bounding_boxes = detector(im)

# if len(bounding_boxes) == 0:
#     print("Лицо не обнаружено")

# else:
#     clm = mpfit.clm.load_dlib_clm_model()

#     result = clm.fit_from_bb(im, bounding_boxes[0], max_iters=50)

#     im.view_landmarks(result.final_shape)

import cv2
import numpy as np
import menpo.io as mio
import menpofit as mpfit
from menpo.shape import bounding_box
import menpofit.clm
print(dir(menpofit.clm.ActiveShapeModel))
# 1. Загрузка изображения с помощью Menpo
im = mio.import_image('C:/IT/Python/Face_ID/Galkin1.jpg')
im_gray = im.as_greyscale()

# 2. Детекция лица с использованием OpenCV Haar Cascade
img_cv = np.array(im_gray.pixels, dtype=np.uint8)

img = cv2.imread('Galkin1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#faces = face_cascade.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=4)
faces = face_cascade.detectMultiScale(gray)

if len(faces) == 0:
    print("Лицо не обнаружено")
else:
    (x, y, w, h) = faces[0]
    #bb = bounding_box(np.array([[x, y], [x + w, y + h]]))
    bb = bounding_box([x, y], [x + w, y + h])

    
    # 3. Загрузка CLM-модели и подгонка
    clm = mpfit.clm.load_dlib_clm_model()
    result = clm.fit_from_bb(im, bb, max_iters=50)
    
    # 4. Визуализация результата
    im.view_landmarks(result.final_shape)
