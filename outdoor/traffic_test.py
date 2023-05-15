import cv2
import os
import time

# Load the pre-trained car classifier
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Set the output folder and delete any existing images
output_folder = 'car_images'
if os.path.exists(output_folder):
    filelist = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]
    for f in filelist:
        os.remove(os.path.join(output_folder, f))
else:
    os.mkdir(output_folder)

# Start the webcam
cap = cv2.VideoCapture(0)

# Set a counter for the images
counter = 0

# Set the end time
end_time = time.time() + 50

# Set a list to store the locations of previously detected cars
previous_cars = []

# Loop through the webcam feed for 20 seconds
while time.time() < end_time:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop through the detected cars and save images
    for (x,y,w,h) in cars:
        # Check if the car is in a location that has not been detected before
        if all(abs(x-xp) > w/2 and abs(y-yp) > h/2 for (xp,yp,wp,hp) in previous_cars):
            # Draw a rectangle around the car
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
            # Save the image of the car
            car_image = frame[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_folder, f'car_{counter}.jpg'), car_image)
            counter += 1
        
            # Add the location of the car to the list of previous cars
            previous_cars.append((x,y,w,h))
    
    # Display the frame
    cv2.imshow('frame', frame)
    
    # Quit the loop if the 'q' key is pressed or time is up
    if cv2.waitKey(1) & 0xFF == ord('q') or time.time() >= end_time:
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
