import cv2

class Creator:
    load_Cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    webcam = cv2.VideoCapture(0)

    def __init__(self):
        self.username = input("Enter Username: ")
        self.unique_id = ''

    def details(self):
        while True:
            try:
                self.unique_id = int(input("Enter a Unique ID (max 5 characters): "))
            except ValueError:
                print("Not a Number")
                continue
            if len(str(self.unique_id)) > 5:
                print("ID longer than 5 Characters")
            else:
                break

    def data(self):
        dataFile = open("Credentials.txt", "a")
        dataFile.write(str(self.unique_id) + " " + self.username + "\n")

        return dataFile

    def pics(self):
        pics = 0
        while True:
            status, cam = Creator.webcam.read()
            grayscale = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
            faces = Creator.load_Cascade.detectMultiScale(cam, 1.3, 5)

            for (x, y, w, h) in faces:
                pics = pics + 1
                cv2.imwrite("DataSets/" + str(self.unique_id) + "." + str(pics) + ".bmp", grayscale)

                cv2.rectangle(cam, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.waitKey(50)
            cv2.imshow('Creating DataSets', cam)

            if (pics >= 50):
                break


c = Creator()
c.details()
c.data()
c.pics()
c.webcam.release()
cv2.destroyAllWindows()
