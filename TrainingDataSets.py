import os, cv2
import numpy as np
from PIL import Image


class Trainer:
    file_path = 'DataSets'
    create_LBPH = cv2.face.LBPHFaceRecognizer_create()

    def __init__(self):
        self.faces = []
        self.users = []

    def img(self):
        paths_of_images = [os.path.join(Trainer.file_path, f) for f in os.listdir(Trainer.file_path)]
        for path_image in paths_of_images:
            face_i = Image.open(path_image).convert('L')
            facearray = np.array(face_i, 'uint8')
            user = int(os.path.split(path_image)[-1].split('.')[0])
            self.faces.append(facearray)
            self.users.append(user)
            x = 55
            w = int(facearray.shape[1] * x / 100)
            h = int(facearray.shape[0] * x / 100)
            d = (w, h)
            resized = cv2.resize(facearray, d)
            cv2.imshow("Training", resized)
            cv2.waitKey(50)
        return self.users, self.faces

    def train(self):
        obj = Trainer.img(self)
        print("Training Starting: ")
        self.users, self.faces = obj
        cv2.destroyAllWindows()
        Trainer.create_LBPH.train(self.faces, np.array(self.users))  # training
        Trainer.create_LBPH.write('Trained_Data/Train_DataSets.yml')  # saving data
        cv2.destroyAllWindows()
        print("Training Finished")


t = Trainer()
t.img()
t.train()
