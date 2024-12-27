import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys


# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def image_cropping(source_folder, images_list, destination_folder):
    """
    Processes a list of images, detects faces, crops the first detected face in each image,
    resizes the cropped face to 256x256 pixels, and saves the cropped images to a destination folder.

    Parameters:
        source_folder (str): The folder containing the source images.
        images_list (list of str): A list of image filenames to process.
        destination_folder (str): The folder where the cropped images will be saved.
    
    Returns:
        None
    """
    # Loop through the provided list of image filenames
    for image_file in images_list:
        image = cv2.imread(source_folder+image_file) # Read the image from the source folder
        print(image.shape)
        print(image_file)

        # Convert the color image to grayscale
        # This step is essential for the face detection algorithm, which works on grayscale images
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image using the face_cascade object
        # Parameters:
        # - scaleFactor: Specifies the reduction factor for image scale during detection
        # - minNeighbors: Defines how many neighbors each candidate rectangle should have to retain it
        # - minSize: Specifies the minimum size of the detected face
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Crop the first detected face
        for (x, y, w, h) in faces:
            # Crop the region of interest corresponding to the detected face
            cropped_face = image[y:y+h, x:x+w]
            # Resize the cropped face to a standard size of 256x256 pixels
            resized_face = cv2.resize(cropped_face, (256, 256))
            break  # Exit after the first face

        # Save the processed (cropped and resized) face to the destination folder
        # The filename remains the same as the original image file
        cv2.imwrite(destination_folder+f'{image_file}', resized_face)

"""-------------------------Image Cropping for JAFFE------------------------------------"""
source_folder = "Jaffe/jaffedbase/"
images_list = os.listdir(source_folder)
destination_folder = 'Jaffe/Jaffe_cropped/'

image_cropping(source_folder=source_folder, images_list=images_list, destination_folder=destination_folder)

sys.exit()

"""----------------------Image Cropping for CK+ dataset------------------------------------"""
# Load the image
source_folder = "C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Research Proposal/Facial Expression/CS229-master/CS229-master/CK+/surprise/"
images_list = os.listdir(source_folder)
destination_folder = 'C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Research Proposal/Facial Expression/CS229-master/CS229-master/CK+/Cropped/surprise/'
image_cropping(source_folder=source_folder, images_list=images_list, destination_folder=destination_folder)
sys.exit()

