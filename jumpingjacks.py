import cv2
import mediapipe as mp

# Define the exercise line coordinates
line_coordinates = [(100, 240), (540, 240)]  # Example line from (100, 240) to (540, 240)

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize MediaPipe Pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables for counting exercises
above_line = False
exercise_count = 0

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
        cv2.line(image, (line_coordinates[0][0], line_coordinates[0][1]), (line_coordinates[1][0], line_coordinates[1][1]), (0, 255, 0), 3)

        # Draw the landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Check if the wrist joint (red) crosses the line and is above it
        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        if wrist.y < line_coordinates[0][1]:
            if not above_line:
                above_line = True
                exercise_count += 1
        else:
            above_line = False

        # Display the exercise count
        cv2.putText(image, f"Exercise Count: {int(exercise_count)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the image
    cv2.imshow("Exercise Counter", image)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

