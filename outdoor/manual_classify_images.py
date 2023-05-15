import cv2
import os
import shutil

# Set the path to the folder containing the images
folder_path = "car_images"

# Set the path to the output file to save the labels
output_file = "labels.txt"

# Define the key codes to label the images
POSITIVE_KEY = ord("p")
NEGATIVE_KEY = ord("n")
QUIT_KEY = 27 # ESC key

# Define the output directories
positive_dir = "positive_images"
negative_dir = "negative_images"

# Create the output directories if they do not exist
os.makedirs(positive_dir, exist_ok=True)
os.makedirs(negative_dir, exist_ok=True)

# Open the output file for writing
with open(output_file, "w") as f:

    # Loop through all the files in the folder
    for file_name in os.listdir(folder_path):

        # Load the image
        img_path = os.path.join(folder_path, file_name)
        img = cv2.imread(img_path)

        if img is not None and img.size > 0:
            # Display the image and wait for key input
            cv2.imshow("Image", img)
            key = cv2.waitKey(0)

            # Label the image and write to the output file
            if key == POSITIVE_KEY:
                label = "positive"
                output_path = os.path.join(positive_dir, file_name)
            elif key == NEGATIVE_KEY:
                label = "negative"
                output_path = os.path.join(negative_dir, file_name)
            elif key == QUIT_KEY:
                break
            else:
                continue

            f.write(f"{img_path} {label}\n")

            # Move the image to the output directory
            shutil.move(img_path, output_path)

# Close the window and release resources
cv2.destroyAllWindows()
