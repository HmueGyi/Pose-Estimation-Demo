import cv2
from ultralytics import YOLO

# 1. Load your trained facial model
# Make sure this points to the weight file trained on the Roboflow project in the image
model = YOLO("best.pt") 

# 2. Open Webcam
cap = cv2.VideoCapture(0)

# Keypoint order based on your Roboflow "Edit face Keypoints" screenshot
# Ensure this matches the exact order you labeled them in Roboflow
KEYPOINT_NAMES = [
    "left_eye", 
    "right_eye", 
    "nose", 
    "left_mouth", 
    "right_mouth"
]

# Skeleton connections based on the lines in your image
SKELETON_CONNECTIONS = [
    (0, 1), # left_eye to right_eye
    (0, 2), # left_eye to nose
    (1, 2), # right_eye to nose
    (2, 3), # nose to left_mouth
    (2, 4), # nose to right_mouth
    (3, 4)  # left_mouth to right_mouth
]

while True:
    success, frame = cap.read()
    if not success:
        break

    # 3. Inference
    results = model(frame, stream=True)

    for r in results:
        # r.plot() handles basic drawing, but we'll overlay manual lines for the skeleton
        annotated_frame = r.plot()

        if r.keypoints is not None:
            # xy shape: [num_faces, 5, 2]
            kpts = r.keypoints.xy.cpu().numpy()
            # confidence scores
            confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None

            for face_idx, face in enumerate(kpts):
                # Draw the skeleton lines
                for start_idx, end_idx in SKELETON_CONNECTIONS:
                    # Check if indices are within bounds and have enough confidence
                    if start_idx < len(face) and end_idx < len(face):
                        if confs is not None and (confs[face_idx][start_idx] < 0.25 or confs[face_idx][end_idx] < 0.25):
                            continue
                        
                        start_pt = (int(face[start_idx][0]), int(face[start_idx][1]))
                        end_pt = (int(face[end_idx][0]), int(face[end_idx][1]))
                        
                        # Only draw if the point isn't [0, 0] (which usually means undetected)
                        if start_pt != (0, 0) and end_pt != (0, 0):
                            cv2.line(annotated_frame, start_pt, end_pt, (0, 255, 0), 2)

    # 4. Display
    cv2.imshow("Facial Keypoint Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()