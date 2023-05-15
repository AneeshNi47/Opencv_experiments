import cv2
import os

# Set the paths to the source, positive, and negative folders
source_folder = "path/to/source/folder"
positive_folder = "path/to/positive/folder"
negative_folder = "path/to/negative/folder"

# Set the paths to the cascade classifiers
car_cascade_path = "path/to/car/cascade/classifier.xml"
bus_cascade_path = "path/to/bus/cascade/classifier.xml"
motorcycle_cascade_path = "path/to/motorcycle/cascade/classifier.xml"

# Load the cascade classifiers
car_cascade = cv2.CascadeClassifier(car_cascade_path)
bus_cascade = cv2.CascadeClassifier(bus_cascade_path)
motorcycle_cascade = cv2.CascadeClassifier(motorcycle_cascade_path)

# Loop through all the files in the source folder
for file_name in os.listdir(source_folder):
    # Read the image
    image_path = os.path.join(source_folder, file_name)
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the objects in the image using each cascade classifier
    cars = car_cascade.detectMultiScale(gray)
    buses = bus_cascade.detectMultiScale(gray)
    motorcycles = motorcycle_cascade.detectMultiScale(gray)

    # Classify the image based on which type of vehicle is detected
    if len(cars) > 0:
        print(f"{file_name} is a car.")
        positive_path = os.path.join(positive_folder, "cars", file_name)
        cv2.imwrite(positive_path, image)
    elif len(buses) > 0:
        print(f"{file_name} is a bus.")
        positive_path = os.path.join(positive_folder, "buses", file_name)
        cv2.imwrite(positive_path, image)
    elif len(motorcycles) > 0:
        print(f"{file_name} is a motorcycle.")
        positive_path = os.path.join(positive_folder, "motorcycles", file_name)
        cv2.imwrite(positive_path, image)
    else:
        print(f"{file_name} is a negative image.")
        negative_path = os.path.join(negative_folder, file_name)
        cv2.imwrite(negative_path, image)

    # Remove the original image from the source folder
    os.remove(image_path)

print("Classification complete.")
