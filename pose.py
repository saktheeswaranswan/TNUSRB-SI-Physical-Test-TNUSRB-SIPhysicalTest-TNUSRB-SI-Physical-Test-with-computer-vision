import cv2
import mediapipe as mp

# Define the exercise line coordinates
line_coordinates = [(100, 200), (500, 200)]  # Example line from (100, 200) to (500, 200)

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize MediaPipe Pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to check if a point is above or below a line
def is_above_line(point, line_start, line_end):
    return (line_end[0] - line_start[0]) * (point[1] - line_start[1]) - (line_end[1] - line_start[1]) * (point[0] - line_start[0]) > 0

# Function to count exercise repetitions
def count_exercises(landmarks):
    exercise_count = 0
    above_line = False

    for landmark in landmarks:
        if is_above_line((landmark.x, landmark.y), line_coordinates[0], line_coordinates[1]):
            if not above_line:
                exercise_count += 1
                above_line = True
        else:
            above_line = False

    return exercise_count

# OpenCV setup
cap = cv2.VideoCapture(0)  # Video capture from default camera (change if necessary)

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

        # Get the landmarks
        landmarks = results.pose_landmarks.landmark

        # Count the exercises
        exercise_count = count_exercises(landmarks)

        # Draw the landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the exercise count
        cv2.putText(image, f"Exercise Count: {exercise_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the image
    cv2.imshow("Exercise Counter", image)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

