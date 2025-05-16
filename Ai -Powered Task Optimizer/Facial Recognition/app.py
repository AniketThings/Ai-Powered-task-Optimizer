import cv2
import time
from deepface import DeepFace

# Open Webcam
cap = cv2.VideoCapture(0)

# Start time
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze facial emotion
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']

        # Display emotion on the frame
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
    except:
        cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Facial Emotion Detection', frame)

    # Close webcam after 10 seconds
    if time.time() - start_time > 30:
        print("Time's up! Closing webcam...")
        break

    # Allow manual exit with 'q'
    if cv2.waitKey(1) == ord('q'):
        print("Manual exit detected!")
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()



