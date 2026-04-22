import cv2
from ultralytics import YOLO

# 1. Load your trained model
model = YOLO("best.pt") 

# 2. Open Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam could not be opened")
    exit()

while True:
    success, frame = cap.read()
    
    if success:
        # 3. Inference
        results = model(frame, stream=True)

        for r in results:
            # Annotated frame (Bounding box နဲ့ confidence score တွေပါမယ်)
            annotated_frame = r.plot()

            if r.keypoints is not None:
                # Keypoints ရဲ့ x, y coordinates များကို ယူခြင်း
                kpts = r.keypoints.xy.cpu().numpy()
                # Confidence scores များကို ယူခြင်း
                confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None

                for person_idx, person_kpts in enumerate(kpts):
                    # Point တစ်ခုချင်းစီကို loop ပတ်ပြီး ဆွဲမယ်
                    for kp_idx, kp in enumerate(person_kpts):
                        x, y = int(kp[0]), int(kp[1])
                        
                        # Point က 0 ဖြစ်နေရင် (Detect မမိရင်) မဆွဲဘူး
                        if x == 0 and y == 0:
                            continue
                            
                        # Confidence စစ်မယ် (0.25 ထက်ကျော်မှ ဆွဲမယ်)
                        if confs is not None and confs[person_idx][kp_idx] < 0.25:
                            continue
                        
                        # Point လေးတွေကို အစိမ်းရောင် အဝိုင်းလေးတွေနဲ့ ပြမယ်
                        cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
                        
                        # Point တစ်ခုချင်းစီရဲ့ index နံပါတ်ကိုပါ ဘေးနားမှာ စာသားနဲ့ပြမယ် (စစ်ဆေးရလွယ်အောင်)
                        cv2.putText(annotated_frame, str(kp_idx), (x + 5, y - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 4. Display Result
        cv2.imshow("Student Behavior Keypoints Only", annotated_frame)

        # 'q' နှိပ်ရင် ပိတ်မယ်
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()