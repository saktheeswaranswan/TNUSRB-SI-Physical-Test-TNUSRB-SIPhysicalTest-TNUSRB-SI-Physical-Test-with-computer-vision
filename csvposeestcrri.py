import cv2
import mediapipe as mp
import csv

# Define the exercise line coordinates
line_coordinates = [(0, 240), (640, 240)]  # Horizontal line at y-coordinate 240

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize MediaPipe Pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables for counting exercises
above_line = False
exercise_count = 0

# Read the pose nodes from the CSV file
pose_nodes = []
with open('pose_nodes.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        pose_nodes.append(row[1])

# OpenCV setup
cap = cv2.VideoCapture("gymsitups.mp4")  # Video capture from default camera (change if necessary)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Unable to read from webcam.")
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Pose
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        # Draw the line on the image
        cv2.line(image, line_coordinates[0], line_coordinates[1], (0, 255, 0), 3)

        # Draw the landmarks and numbers on the image
        annotated_image = image.copy()
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            # Get the coordinates of the landmark
            height, width, _ = image.shape
            cx, cy = int(landmark.x * width), int(landmark.y * height)

            # Check if the landmark index is within the provided node points list
            if idx < len(pose_nodes):
                # Check if the landmark crosses the line from bottom to top
                if cy > line_coordinates[0][1]:
                    if not above_line:
                        above_line = True
                else:
                    if above_line:
                        above_line = False
                        exercise_count += 1

                # Draw the landmark and number on the image
                cv2.circle(annotated_image, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(annotated_image, str(exercise_count), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated_image, f"{pose_nodes[idx]}", (cx, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Display the exercise count
        cv2.putText(annotated_image, f"Exercise Count: {exercise_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the annotated image
        cv2.imshow("Exercise Counter", annotated_image)

    else:
        # Show the original image if no pose landmarks are detected
        cv2.imshow("Exercise Counter", image)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

