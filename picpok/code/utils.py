import os
import shutil
import glob
import re
import PIL
import numpy as np
import pickle

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

def reduce_image_size(unknown_image,size):
    if max(unknown_image.shape) > size:
        pil_img = PIL.Image.fromarray(unknown_image)
        pil_img.thumbnail((size, size), PIL.Image.LANCZOS)
        unknown_image = np.array(pil_img)
    return unknown_image

def save_obj(obj, name ):
    with open('../'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('../' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def sort_by_person(picture_character_dict,path_to_folder):
    path_to_folder = path_to_folder + "/"
    for picture in picture_character_dict:
        if len(picture_character_dict[picture]):
            dest_folder = path_to_folder
            for character in picture_character_dict[picture]:
                dest_folder += str(character)+"_"
            if not os.path.exists(dest_folder):
                os.mkdir(dest_folder)
                 #TODO tidy up, assuming only one file extension
                shutil.move(glob.glob(path_to_folder + picture+"*")[0], dest_folder)
            else:
                shutil.move(glob.glob(path_to_folder + picture+"*")[0], dest_folder)

def collapse_folders(path_to_folder):
    for folder in next(os.walk(path_to_folder))[1]:
        files = os.listdir(path_to_folder+"/"+folder)
        for f in files:
            shutil.move(path_to_folder+"/"+folder+"/"+f, path_to_folder)
        os.removedirs(path_to_folder+"/"+folder)