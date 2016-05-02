#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image
import sys


class FaceRecognizer:

    # Path to the Yale Dataset
    path = './yalefaces'
    # For face detection we will use the Haar Cascade provided by OpenCV.
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    # For face recognition we will the the LBPH Face Recognizer 
    #recognizer = cv2.createLBPHFaceRecognizer()
    recognizer = cv2.createFisherFaceRecognizer()

    def get_images_and_labels(self, path):
        # Append all the absolute image paths in a list image_paths
        # We will not read the image with the .sad extension in the training set
        # Rather, we will use them to test our accuracy of the training
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        # images will contains face images
        images = []
        # labels will contains the label that is assigned to the image
        labels = []
        for image_path in image_paths:
            # Read the image and convert to grayscale
            image_pil = Image.open(image_path).convert('L')
            # Convert the image format into numpy array
            image = np.array(image_pil, 'uint8')
            # Get the label of the image
            nbr = int(self.convert_label(os.path.split(image_path)[1].split(".")[1]))
            if nbr == -1:
                continue
            # Detect the face in the image
            faces = self.faceCascade.detectMultiScale(image)
            # If face is detected, append the face to images and the label to labels
            for (x, y, w, h) in faces:
                orig = image[y: y + h, x: x + w]
                res = cv2.resize(orig,(200, 200), interpolation = cv2.INTER_CUBIC)
                images.append(res)
                labels.append(nbr)
                #cv2.imshow("Adding faces to traning set...", res)
                #cv2.waitKey(50)
        # return the images list and labels list
        return images, labels

    def convert_label(self, emo):
        if emo == "normal":
            return 0
        elif emo == "happy":
            return 1
        elif emo == "sad":
            return 2
        elif emo == "surprised":
            return 3
        elif emo == "sleepy":
            return 4
        elif emo == "wink":
            return 5
        else:
            return -1


    def getClassification(self):


        # Call the get_images_and_labels function and get the face images and the 
        # corresponding labels
        images, labels = self.get_images_and_labels(FaceRecognizer.path)
        cv2.destroyAllWindows()

        test_img = images.pop()
        test_label = labels.pop()
        # Perform the tranining
        self.recognizer.train(images, np.array(labels))

        # Append the images with the extension .sad into image_paths
        faces = self.faceCascade.detectMultiScale(test_img)
        nbr_predicted, conf = self.recognizer.predict(test_img)
        nbr_actual = test_label
        if nbr_actual == nbr_predicted:
            print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
        else:
            print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)
        #cv2.imshow("Recognizing Face", test_img)
        #cv2.waitKey(1000)


        image_real = Image.open('who.jpg').convert('L')
        # Convert the image format into numpy array
        image = np.array(image_real, 'uint8')

        #cv2.imshow("real face", image)
        #cv2.waitKey(5000)
        # Detect the face in the image
        faces = self.faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            orig = image[y: y + h, x: x + w]
            res = cv2.resize(orig,(200, 200), interpolation = cv2.INTER_CUBIC)
            nbr_predicted, conf = self.recognizer.predict(res)
            print nbr_predicted
            return nbr_predicted
            #cv2.imshow("real face", res)
            #cv2.waitKey(5000)
