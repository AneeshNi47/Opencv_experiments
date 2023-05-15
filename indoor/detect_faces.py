import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)

# remove previous files from the faces folder
if not os.path.exists('faces'):
    os.makedirs('faces')
else:
    for image in os.listdir('faces/'):
        os.remove(f'faces/{image}')

existing_faces = []

while True:
    # Capture frame from webcam
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break
        
    # Convert to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)

    for x, y, w, h in faces:
        current_face = gray_img[y:y+h, x:x+w]
        current_face_filename = f"{x}-{y}.png"

        if current_face_filename not in existing_faces:
            existing_faces.append(current_face_filename)
            cv2.imwrite(f'faces/{current_face_filename}', current_face)

    # Display video feed with bounding boxes around detected faces
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Live Face Detection", frame)        

    # Exit loop by pressing "q"
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
