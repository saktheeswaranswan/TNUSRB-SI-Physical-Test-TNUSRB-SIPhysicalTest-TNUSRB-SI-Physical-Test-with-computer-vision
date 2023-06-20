import cv2
import mediapipe as mp
import csv

# Define the exercise line coordinates
line_coordinates = [(100, 240), (540, 240)]  # Example line from (100, 240) to (540, 240)

# Define the intersection lines (horizontal and vertical)
intersection_lines = [
    [(100, 200), (540, 200)],   # Example horizontal line
    [(320, 100), (320, 480)]    # Example vertical line
]

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize MediaPipe Pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables for counting exercises
above_lines = [False] * len(intersection_lines)
exercise_counts = [0] * len(intersection_lines)

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
        # Draw the lines on the image
        for line in line_coordinates:
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            cv2.line(image, pt1, pt2, (0, 255, 0), 3)

        # Draw the intersection lines on the image
        for line in intersection_lines:
            pt1 = (line[0][0], line[0][1])
            pt2 = (line[1][0], line[1][1])
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)

        # Draw the landmarks and numbers on the image
        annotated_image = image.copy()
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            # Get the coordinates of the landmark
            height, width, _ = image.shape
            cx, cy = int(landmark.x * width), int(landmark.y * height)

            # Check if the landmark index is within the provided node points list
            if idx < len(pose_nodes):
                # Check if the landmark crosses any of the intersection lines
                for i, line in enumerate(intersection_lines):
                    if (cy < line[0][1] and line[0][0] <= cx <= line[1][0]) or \
                            (cy > line[0][1] and line[0][0] >= cx >= line[1][0]):
                        if not above_lines[i]:
                            above_lines[i] = True
                            exercise_counts[i] += 1
                    else:
                        above_lines[i] = False

                # Draw the landmark and number on the image
                cv2.circle(annotated_image, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(annotated_image, str(exercise_counts[i]), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated_image, f"{pose_nodes[idx]}", (cx, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1, cv2.LINE_AA)

        # Display the exercise counts for each line
        for i, count in enumerate(exercise_counts):
            cv2.putText(annotated_image, f"Line {i+1} Count: {count}", (10, 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

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

