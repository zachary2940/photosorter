import face_recogniser
import dbscan
import utils
import numpy as np
import time
import pprint


# Load the jpg file into a numpy array
if __name__=="__main__":
    fr = face_recogniser.Face_Recogniser()
    # fia = fr.create_face_image_array("../pictures")
    # fr.save_face_image_array(fia)

    # known_names,known_face_encodings = fr.save_face_encodings("../faces")
    # fr.test_images("../pictures", known_names, known_face_encodings, tolerance=0.6, show_distance=False)

    # known_face_image_array,known_face_encodings = fr.save_unique_face_encodings("../pictures",[],[])
    # fr.save_face_image_array(known_face_image_array)

    # s=time.time()
    # picture_encoding_dict = fr.save_face_encodings_dictionary("../pictures")
    # print("time taken:{}".format((time.time()-s)*1000))
    # utils.save_obj(picture_encoding_dict,"picture_encoding_dict")

    # picture_encoding_dict = utils.load_obj("picture_encoding_dict")

    # # I want a dictionary with characters as keys and photos as values
    # # currently I can get clusters but I need to assign them an index
    # # Then these points must also relate to their photos so I can get them as values

    # db_clusterer = dbscan.DB_Clustering(epsilon=0.3,min_samples=3)
    # # X = np.array([[1, 2], [2, 2], [2, 3],
    # #...               [8, 7], [8, 8], [25, 80]])
    # encodings=[]
    # for picture in picture_encoding_dict:
    #     for encoding in picture_encoding_dict[picture]:
    #             encodings.append(encoding)
    # db_clusterer.add_training_set(encodings)

    # labels = db_clusterer.train()

    # utils.save_obj(db_clusterer,"dbscan_model")
    # db_clusterer = utils.load_obj("dbscan_model")

    # picture_character_dict = db_clusterer.match_picture_to_character(picture_encoding_dict)
    # utils.save_obj(picture_character_dict,"picture_character_dict")

    # picture_character_dict = utils.load_obj("picture_character_dict")
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(picture_character_dict)

    # utils.sort_by_person(picture_character_dict,"D:/Projects/Python_projects/facial_recognition/picpok/pictures")

    utils.collapse_folders("D:/Projects/Python_projects/facial_recognition/picpok/pictures")
