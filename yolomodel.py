import cv2
from ultralytics import YOLO

# YOLO 모델 불러오기
model = YOLO('yolov8n-pose.pt')  # YOLOv8 포즈 모델

# 비디오에서 keypoints 추출하는 함수 (디버깅용으로 출력 추가)
def extract_keypoints_from_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO로 프레임에서 포즈 추출
        results = model(frame)
        
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()  # NumPy 배열로 변환
                keypoints_sequence.append(keypoints.flatten())  # 1D 벡터로 변환
                print(f"Keypoints: {keypoints}")  # keypoints 출력
            else:
                print("No keypoints detected in this frame.")
        
    cap.release()
    return keypoints_sequence

# 비디오 경로 설정
video_path = 'datasets/video.mp4'
extract_keypoints_from_video(video_path, model)
