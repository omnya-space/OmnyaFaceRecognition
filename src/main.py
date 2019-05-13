import cv2
import os
from image_processor import label_image
from face_recognizer import recognize_faces
from admin import register_faces

if __name__ == '__main__':

    register_faces('faces')

    test_image_path = 'test/beckham2.jpg'
    image=cv2.imread(test_image_path)
    people=recognize_faces(test_image_path)
    if people is not None:
        for person in people:
            name = person[0]
            bounding_box = person[1]
            image = label_image(image, name, bounding_box)

        cv2.imwrite(os.path.join('labelled/', '%s.png') % name, image)



