import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        # Try face detection
        analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False, detector_backend="retinaface")
        print("Face detected! Emotion:", analysis[0]['dominant_emotion'])
    except:
        print("No Face Detected!")

    cv2.imshow('Face Detection Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
