import cv2, pyttsx3, datetime


class Recognition:
    speak = pyttsx3.init()  # initialising engine
    load_Cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    webcam = cv2.VideoCapture(0)
    create_LBPH = cv2.face.LBPHFaceRecognizer_create()
    trained_data = create_LBPH.read("Trained_Data/Train_DataSets.yml")
    name = ""

    def __init__(self):
        self.user = {}
        self.name2 = ''

    def open_data_text(self):
        global openTextFile
        openTextFile = open("Credentials.txt", "r")
        return openTextFile

    def text_file(self):
        for c in openTextFile:
            a, b = c.split(" ")
            self.user[a] = b.replace("\n", "")

    def id_output(self):
        current_time = datetime.datetime.now()
        who_at_door = open("WhosAtTheDoor.txt", "a")
        while Recognition.name != "Success":
            status, cam = Recognition.webcam.read()
            grayscale = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
            faces = Recognition.load_Cascade.detectMultiScale(grayscale, 1.3, 5)
            cv2.imshow("Recognition", cam)
            cv2.waitKey(50)

            for (x, y, w, h) in faces:
                cv2.rectangle(cam, (x, y), (x + w, y + h), (255, 0, 0), 2)
                unique_id, conf = Recognition.create_LBPH.predict(
                    grayscale[y:y + h, x:x + w])  # returns id and confidence level
                print(conf)

                if conf > 65:
                    Recognition.name = "Unknown"
                    Recognition.speak.say(Recognition.name)
                    Recognition.speak.runAndWait()
                    who_at_door.write(str(Recognition.name) + " is at the door "
                                          + current_time.strftime(" %Y-%m-%d %H:%M:%S\n"))
                elif conf < 65:
                    Recognition.name = "Success"
                    self.name2 = "Welcome back" + self.user[str(unique_id)]
                    Recognition.speak.say(self.name2)
                    Recognition.speak.runAndWait()
                    who_at_door.write(str(self.user[str(unique_id)] + " is at the door "
                                          + current_time.strftime(" %Y-%m-%d %H:%M:%S\n")))
                    break


r = Recognition()
r.open_data_text()
r.text_file()
r.id_output()
Recognition.webcam.release()
cv2.destroyAllWindows()
