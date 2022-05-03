# Facial-Recognition-with-Voice-Output-and-Log-Filing

**CreatingDataSets.py**

This file is responsible for capturing images from the input image(webcam). In this case it captures 50 images and converts it into grayscale. In addition it asks for username and ID which is saved to an external file called ```Credentials.txt```

**TrainingDataSets.py**

This takes the images captured from ```CreatingDataSets.py``` and trains the model using a Local Binary Pattern Histogram (LBPH) classifier and the ```haarcascade_frontalface_default.xml``` pre-trained face detection model.

**Recognition.py**

This is where the facial recognition happens. It compares the similarity of pre-existing histogram with a new histogram and if the outcome of ```conf``` is below 65, the face will be recognised. Once the face is recognised it will save the event at ```WhosAtTheDoor.txt``` with the user ID as well as the time and date

