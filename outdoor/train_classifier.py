import cv2

# Set the paths to the positive and negative image directories
positive_images_path = 'path/to/positive/images'
negative_images_path = 'path/to/negative/images'

# Set the path to the output directory for the classifier
output_directory = 'path/to/output/directory'

# Set the size of the object you want to detect
object_size = (50, 50)

# Set the number of stages for the classifier
num_stages = 10

# Set the other parameters for the classifier
params = cv2.CascadeClassifierParams()
params.minHitRate = 0.995
params.maxFalseAlarmRate = 0.5
params.featureType = cv2.HaarFeatureParams_LBP

# Create the positive samples file
positive_samples_file = 'positive_samples.txt'
command = f'opencv_createsamples -info {positive_samples_file} -vec positive_samples.vec -w {object_size[0]} -h {object_size[1]}'
os.system(command)

# Train the classifier
command = f'opencv_traincascade -data {output_directory} -vec positive_samples.vec -bg {negative_images_path} -numPos 1000 -numNeg 2000 -w {object_size[0]} -h {object_size[1]} -numStages {num_stages} -maxFalseAlarmRate 0.5 -featureType LBP'
os.system(command)

# Load the trained classifier
classifier = cv2.CascadeClassifier(f'{output_directory}/cascade.xml')

# Test the classifier on a test image
test_image = cv2.imread('test_image.jpg')
gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
objects = classifier.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=object_size)

# Draw bounding boxes around the detected objects
for (x, y, w, h) in objects:
    cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the test image with the detected objects
cv2.imshow