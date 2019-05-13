import cv2
import numpy as np
from face_detector import FaceDetector
from model import Model
import mxnet as mx
import pickle


with open('face_vectors.pickle', 'rb') as handle:
    face_vectors = pickle.load(handle)
detector_path="mtcnn-model"
face_detector=FaceDetector(model_folder=detector_path, ctx=mx.cpu(0), num_worker=1, accurate_landmark = True, threshold=[0.6, 0.7, 0.8])
model = Model()


def recognize_faces(test_image_path):
    people_in_picture=[]
    image=cv2.imread(test_image_path)
    results = face_detector.get_points_and_bbox(image)
    if results is None:
        print ("No person detected")
        return None

    for result in results:
        points=result[0]
        vector_to_test = model.get_vector(points)
        bounding_box=result[1]
        maxSimilarity=0
        similarityThreshold=0.4
        for name, vector in face_vectors.items():
            similarity = np.dot(vector_to_test, vector.T)
            if similarity > maxSimilarity:
                maxSimilarity = similarity
                best_matched_name=name
        if maxSimilarity > similarityThreshold:
            print ("Recognized ", best_matched_name)
            people_in_picture.append([best_matched_name, bounding_box])
    return people_in_picture





