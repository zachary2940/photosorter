import utils
import face_recognition.api as face_recognition
import multiprocessing
import itertools
import sys
import PIL.Image
import numpy as np
import os

class Face_Recogniser:
    def __init__(self):
        pass
    
    def save_unique_faces_model(self, encodings):

        pass


    # euclidean distance is not the best way
    def save_unique_faces_euclidean_distance(self,folder,known_face_encodings,known_face_image_array,tolerance=0.5):
        for file in utils.image_files_in_folder(folder):
            basename = os.path.splitext(os.path.basename(file))[0]
            img = face_recognition.load_image_file(file)
            encodings = face_recognition.face_encodings(img)
            for i,encoding in enumerate(encodings):
                distances = face_recognition.face_distance(known_face_encodings, encoding)
                if distances.size == 0 :
                    print("no distance found")
                    min_distance=1000
                else:
                    min_distance = np.min(distances)
                if min_distance<=tolerance:
                    print("recognised")
                    index_recognised_face = np.argwhere(distances==min_distance)
                else:
                    print("Unknown")
                    face_locations = face_recognition.face_locations(img)
                    top, right, bottom, left = face_locations[i]
                    known_face_image_array.append(img[top:bottom, left:right])
                    known_face_encodings.append(encoding)
        
        return known_face_image_array,known_face_encodings


    def create_face_image_array(self, folder):
        face_image_array=[]
        for file in utils.image_files_in_folder(folder):
            # basename = os.path.splitext(os.path.basename(file))[0]
            image = face_recognition.load_image_file(file)
            image = utils.reduce_image_size(image,1200)
            face_locations = face_recognition.face_locations(image)
            if len(face_locations)>0: 
                for face_location in face_locations:
                    top, right, bottom, left = face_location
                    face_image_array.append(image[top:bottom, left:right])
            else:
                print("No face detected")
        return face_image_array

    def save_face_image_array(self,face_image_array):
        for i,face_image in enumerate(face_image_array):
            pil_image = PIL.Image.fromarray(face_image)
            pil_image.save("../faces/{}.png".format(i))

    def save_face_encodings_dictionary(self, folder):
        picture_encoding_dict = {}

        for file in utils.image_files_in_folder(folder):
            basename = os.path.splitext(os.path.basename(file))[0]
            img = face_recognition.load_image_file(file)
            encodings = face_recognition.face_encodings(img)
            picture_encoding_dict[basename]=encodings
        return picture_encoding_dict

