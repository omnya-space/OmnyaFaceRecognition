import cv2
from face_detector import FaceDetector
from model import Model
import mxnet as mx
import os
import pickle



detector_path="mtcnn-model"
face_detector=FaceDetector(model_folder=detector_path, ctx=mx.cpu(0), num_worker=1, accurate_landmark = True, threshold=[0.6, 0.7, 0.8])
model = Model()



def register_faces(faces_path):
    face_vectors = {}
    for root, dirs, files in os.walk(faces_path):
        for filename in files:
            name=filename.split(".")[0]
            image=cv2.imread(faces_path+"/"+filename)
            points = face_detector.get_points_and_bbox(image)[0][0]
            vector = model.get_vector(points)
            face_vectors[name]=vector
            print ("Registered ",name)
    with open('face_vectors.pickle', 'wb') as handle:
        pickle.dump(face_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)



