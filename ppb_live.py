import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib


protoFile_mpi = "openpose-master/models/pose/body_25/pose_deploy.prototxt"
weightsFile_mpi = "openpose-master/models/pose/body_25/pose_iter_584000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile_mpi, weightsFile_mpi)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

BODY_PARTS_MPI = {
    0: "Head",
    1: "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "RHip",
    9: "RKnee",
    10: "RAnkle",
    11: "LHip",
    12: "LKnee",
    13: "LAnkle",
    14: "Chest",
    15: "Background"
}


model = joblib.load('trained_model.pkl')

def extract_keypoints(frame, net):
    image_height = 1056
    image_width = 720

    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(input_blob)
    out = net.forward()
    
    frame_height, frame_width = frame.shape[:2]
    points = []
    for i in range(len(BODY_PARTS_MPI)):
        prob_map = out[0, i, :, :]
        _, _, _, point = cv2.minMaxLoc(prob_map)
        
        x = (frame_width * point[0]) / out.shape[3]
        y = (frame_height * point[1]) / out.shape[2]
        points.append((x, y))
    return points

def prepare_data_for_prediction(keypoints):
    relevant_parts = ["Head", "Neck", "RShoulder", "LShoulder"]
    indices = [list(BODY_PARTS_MPI.keys())[list(BODY_PARTS_MPI.values()).index(part)] for part in relevant_parts]

    data = []
    for idx in indices:
        data.append(keypoints[idx][0])  
        data.append(keypoints[idx][1])  
    
    rshoulder_idx = list(BODY_PARTS_MPI.keys())[list(BODY_PARTS_MPI.values()).index("RShoulder")]
    lshoulder_idx = list(BODY_PARTS_MPI.keys())[list(BODY_PARTS_MPI.values()).index("LShoulder")]

    shoulder_vector = np.array(keypoints[rshoulder_idx]) - np.array(keypoints[lshoulder_idx])
    shoulder_angle = np.arctan2(shoulder_vector[1], shoulder_vector[0]) * 180 / np.pi
    data.append(shoulder_angle)

    return np.array([data])


def draw_keypoints(frame, keypoints):
    essential_keypoints = [keypoints[i] for i in [0, 1, 2, 5]]
    for point in essential_keypoints:
        x, y = point
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    keypoints = extract_keypoints(frame, net)
    frame_with_keypoints = draw_keypoints(frame, keypoints)
    data = prepare_data_for_prediction(keypoints)
    prediction = model.predict(data)
    
    text = "Asymmetric" if prediction[0] == 1 else "Symmetric"
    color = (0, 0, 255) if text == "Asymmetric" else (255, 0, 0)
    cv2.putText(frame_with_keypoints, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Real-time Shoulder Asymmetry Detection', frame_with_keypoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()