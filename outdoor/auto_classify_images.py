import cv2
import os

# Set the paths to the source, positive, and negative folders
source_folder = "car_images"
positive_folder = "positive_images"
negative_folder = "positive_images"

# Set the path to the cascade classifier
cascade_path = "haarcascade_car.xml"

# Load the cascade classifier
cascade = cv2.CascadeClassifier(cascade_path)

# Loop through all the files in the source folder
for file_name in os.listdir(source_folder):
    # Read the image
    image_path = os.path.join(source_folder, file_name)
    image = cv2.imread(image_path)

    if image is not None and image.size > 0:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect the objects in the image
        objects = cascade.detectMultiScale(gray)

        # If objects are detected, move the image to the positive folder
        if len(objects) > 0:
            print(f"{file_name} is a positive image.")
            positive_path = os.path.join(positive_folder, file_name)
            cv2.imwrite(positive_path, image)
            os.remove(image_path)
        # Otherwise, move the image to the negative folder
        else:
            print(f"{file_name} is a negative image.")
            negative_path = os.path.join(negative_folder, file_name)
            cv2.imwrite(negative_path, image)
            os.remove(image_path)

print("Classification complete.")
