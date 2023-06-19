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

        # Draw the landmarks and numbers on the image
        annotated_image = image.copy()
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            # Get the coordinates of the landmark
            height, width, _ = image.shape
            cx, cy = int(landmark.x * width), int(landmark.y * height)

            # Check if landmark 13 (right wrist) crosses the line
            if idx == mp_pose.PoseLandmark.RIGHT_WRIST.value:
                if cy < line_coordinates[0][1]:
                    if not above_line:
                        above_line = True
                        exercise_count += 1
                else:
                    above_line = False

                # Draw the landmark and number on the image
                cv2.circle(annotated_image, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(annotated_image, str(exercise_count), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)

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

