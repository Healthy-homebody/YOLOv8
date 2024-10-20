# # # # # import numpy as np
# # # # # import pandas as pd
# # # # # from dtaidistance import dtw

# # # # # a = [1, 2, 5, 7, 4, 3, 6, 8, 2, 1]
# # # # # b = [3, 6, 1, 2, 8, 9, 3, 4, 3, 2, 1, 2]
# # # # # c = [2, 5, 7, 4, 3, 6, 8, 2, 1, 1]

# # # # import cv2
# # # # import numpy as np
# # # # from ultralytics import YOLO
# # # # from dtaidistance import dtw

# # # # # YOLO 모델 불러오기
# # # # model = YOLO('yolov8n-pose.pt')  # YOLOv8 포즈 모델 경로

# # # # # 비디오에서 keypoints 추출하는 함수
# # # # def extract_keypoints_from_video(video_path, model):
# # # #     cap = cv2.VideoCapture(video_path)
# # # #     keypoints_sequence = []
# # # #     max_keypoints = 34  # Keypoints 배열의 고정된 크기 (17개의 keypoints, 각 2D 좌표)

# # # #     while cap.isOpened():
# # # #         ret, frame = cap.read()
# # # #         if not ret:
# # # #             break
        
# # # #         # YOLO로 프레임에서 포즈 추출
# # # #         results = model(frame)
        
# # # #         for result in results:
# # # #             if result.keypoints is not None:
# # # #                 # Keypoints 추출 (xy 좌표만 사용)
# # # #                 keypoints = result.keypoints.xy.cpu().numpy()  # NumPy 배열로 변환
# # # #                 xy_keypoints = keypoints.flatten()  # 1D로 평탄화
                
# # # #                 # Keypoints 배열의 크기를 고정 (34로 맞춤, 부족하면 0으로 패딩)
# # # #                 if len(xy_keypoints) < max_keypoints:
# # # #                     padded_keypoints = np.zeros(max_keypoints)
# # # #                     padded_keypoints[:len(xy_keypoints)] = xy_keypoints
# # # #                     keypoints_sequence.append(padded_keypoints)
# # # #                 else:
# # # #                     keypoints_sequence.append(xy_keypoints[:max_keypoints])
# # # #             else:
# # # #                 # keypoints가 감지되지 않으면 0으로 패딩된 벡터 추가
# # # #                 keypoints_sequence.append(np.zeros(max_keypoints))
        
# # # #     cap.release()

# # # #     # keypoints_sequence를 배열로 변환
# # # #     keypoints_sequence = np.array(keypoints_sequence)

# # # #     return keypoints_sequence

# # # # # 두 시퀀스 간의 DTW 거리 계산
# # # # def calculate_dtw_distance(seq1, seq2):
# # # #     # 확인: 두 시퀀스의 모양
# # # #     print(f"Shape of seq1: {seq1.shape}")
# # # #     print(f"Shape of seq2: {seq2.shape}")

# # # #     # 각 시퀀스를 2D로 변환 (프레임 개수는 다를 수 있음)
# # # #     min_len = min(len(seq1), len(seq2))
# # # #     seq1_flat = seq1[:min_len]  # 두 시퀀스의 길이를 동일하게 맞춤
# # # #     seq2_flat = seq2[:min_len]

# # # #     # 프레임별로 DTW 거리 계산 (1차원 배열로 처리)
# # # #     distances = []
# # # #     for i in range(min_len):
# # # #         distance = dtw.distance(seq1_flat[i], seq2_flat[i])
# # # #         distances.append(distance)

# # # #     # 전체 평균 DTW 거리 반환
# # # #     return np.mean(distances)

# # # # # 두 영상의 유사도를 계산하는 메인 함수
# # # # def compare_videos(video_path1, video_path2, model):
# # # #     # 두 비디오에서 keypoints 시퀀스를 추출
# # # #     keypoints_seq1 = extract_keypoints_from_video(video_path1, model)
# # # #     keypoints_seq2 = extract_keypoints_from_video(video_path2, model)

# # # #     # 두 시퀀스 간의 DTW 거리 계산
# # # #     dtw_distance = calculate_dtw_distance(keypoints_seq1, keypoints_seq2)
    
# # # #     # 유사도를 계산 (거리가 작을수록 더 유사함)
# # # #     print(f"DTW Distance between the two videos: {dtw_distance}")

# # # # # 예시 실행
# # # # video1_path = 'datasets/video6.mp4'  # 첫 번째 영상 경로
# # # # video2_path = 'datasets/video61.mp4'  # 두 번째 영상 경로

# # # # compare_videos(video1_path, video2_path, model)

# # # # # 결과: 두 영상의 유사도를 DTW 거리로 계산하여 출력

# # # import cv2
# # # import numpy as np
# # # from ultralytics import YOLO
# # # from dtaidistance import dtw

# # # # YOLO 모델 불러오기
# # # model = YOLO('yolov8n-pose.pt')  # YOLOv8 포즈 모델 경로

# # # # 비디오에서 keypoints 추출하는 함수 + 화면에 그리기
# # # def extract_keypoints_and_display_video(video_path, model):
# # #     cap = cv2.VideoCapture(video_path)
# # #     keypoints_sequence = []
# # #     max_keypoints = 34  # Keypoints 배열의 고정된 크기 (17개의 keypoints, 각 2D 좌표)
    
# # #     while cap.isOpened():
# # #         ret, frame = cap.read()
# # #         if not ret:
# # #             break
        
# # #         # YOLO로 프레임에서 포즈 추출
# # #         results = model(frame)

# # #         for result in results:
# # #             if result.keypoints is not None:
# # #                 # Keypoints 추출 (xy 좌표만 사용)
# # #                 keypoints = result.keypoints.xy.cpu().numpy()  # NumPy 배열로 변환
# # #                 xy_keypoints = keypoints.flatten()  # 1D로 평탄화
                
# # #                 # Keypoints 배열의 크기를 고정 (34로 맞춤, 부족하면 0으로 패딩)
# # #                 if len(xy_keypoints) < max_keypoints:
# # #                     padded_keypoints = np.zeros(max_keypoints)
# # #                     padded_keypoints[:len(xy_keypoints)] = xy_keypoints
# # #                     keypoints_sequence.append(padded_keypoints)
# # #                 else:
# # #                     keypoints_sequence.append(xy_keypoints[:max_keypoints])
                
# # #                 # Keypoints 시각적으로 프레임에 그리기
# # #                 for i in range(0, len(xy_keypoints), 2):
# # #                     x, y = int(xy_keypoints[i]), int(xy_keypoints[i + 1])
# # #                     cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 녹색 원으로 keypoint 그리기
                
# # #         # 화면에 프레임 보여주기
# # #         cv2.imshow('Keypoints on Video', frame)
        
# # #         # 'q' 키를 누르면 종료
# # #         if cv2.waitKey(1) & 0xFF == ord('q'):
# # #             break
    
# # #     cap.release()
# # #     cv2.destroyAllWindows()

# # #     # keypoints_sequence를 배열로 변환
# # #     keypoints_sequence = np.array(keypoints_sequence)

# # #     return keypoints_sequence

# # # # 두 시퀀스 간의 DTW 거리 계산
# # # def calculate_dtw_distance(seq1, seq2):
# # #     # 확인: 두 시퀀스의 모양
# # #     print(f"Shape of seq1: {seq1.shape}")
# # #     print(f"Shape of seq2: {seq2.shape}")

# # #     # 각 시퀀스를 2D로 변환 (프레임 개수는 다를 수 있음)
# # #     min_len = min(len(seq1), len(seq2))
# # #     seq1_flat = seq1[:min_len]  # 두 시퀀스의 길이를 동일하게 맞춤
# # #     seq2_flat = seq2[:min_len]

# # #     # 프레임별로 DTW 거리 계산 (1차원 배열로 처리)
# # #     distances = []
# # #     for i in range(min_len):
# # #         distance = dtw.distance(seq1_flat[i], seq2_flat[i])
# # #         distances.append(distance)

# # #     # 전체 평균 DTW 거리 반환
# # #     return np.mean(distances)

# # # # 두 영상의 유사도를 계산하는 메인 함수
# # # def compare_videos(video_path1, video_path2, model):
# # #     # 첫 번째 비디오에서 keypoints 시퀀스를 추출하고 화면에 표시
# # #     keypoints_seq1 = extract_keypoints_and_display_video(video_path1, model)
# # #     # 두 번째 비디오에서 keypoints 시퀀스를 추출하고 화면에 표시
# # #     keypoints_seq2 = extract_keypoints_and_display_video(video_path2, model)

# # #     # 두 시퀀스 간의 DTW 거리 계산
# # #     dtw_distance = calculate_dtw_distance(keypoints_seq1, keypoints_seq2)
    
# # #     # 유사도를 계산 (거리가 작을수록 더 유사함)
# # #     print(f"DTW Distance between the two videos: {dtw_distance}")

# # # # 예시 실행
# # # video1_path = 'datasets/video6.mp4'  # 첫 번째 영상 경로
# # # video2_path = 'datasets/video61.mp4'  # 두 번째 영상 경로

# # # compare_videos(video1_path, video2_path, model)

# # import cv2
# # import numpy as np
# # from ultralytics import YOLO
# # from dtaidistance import dtw

# # # YOLO 모델 불러오기
# # model = YOLO('yolov8n-pose.pt')  # YOLOv8 포즈 모델 경로

# # # keypoints 좌표를 [0, 1]로 정규화하는 함수
# # def normalize_keypoints(keypoints, frame_width, frame_height):
# #     # x 좌표는 프레임 너비로, y 좌표는 프레임 높이로 나눠서 정규화
# #     normalized_keypoints = np.copy(keypoints)
# #     for i in range(0, len(keypoints), 2):
# #         normalized_keypoints[i] = keypoints[i] / frame_width  # x 좌표
# #         normalized_keypoints[i + 1] = keypoints[i + 1] / frame_height  # y 좌표
# #     return normalized_keypoints

# # # 비디오에서 keypoints 추출하는 함수 (시각적으로 확인 가능)
# # def extract_keypoints_and_display_video(video_path, model):
# #     cap = cv2.VideoCapture(video_path)
# #     keypoints_sequence = []
# #     max_keypoints = 34  # Keypoints 배열의 고정된 크기 (17개의 keypoints, 각 2D 좌표)
# #     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
# #     while cap.isOpened():
# #         ret, frame = cap.read()
# #         if not ret:
# #             break
        
# #         # YOLO로 프레임에서 포즈 추출
# #         results = model(frame)

# #         for result in results:
# #             if result.keypoints is not None:
# #                 # Keypoints 추출 (xy 좌표만 사용)
# #                 keypoints = result.keypoints.xy.cpu().numpy()  # NumPy 배열로 변환
# #                 xy_keypoints = keypoints.flatten()  # 1D로 평탄화
                
# #                 # 좌표 정규화
# #                 normalized_keypoints = normalize_keypoints(xy_keypoints, frame_width, frame_height)

# #                 # Keypoints 배열의 크기를 고정 (34로 맞춤, 부족하면 0으로 패딩)
# #                 if len(normalized_keypoints) < max_keypoints:
# #                     padded_keypoints = np.zeros(max_keypoints)
# #                     padded_keypoints[:len(normalized_keypoints)] = normalized_keypoints
# #                     keypoints_sequence.append(padded_keypoints)
# #                 else:
# #                     keypoints_sequence.append(normalized_keypoints[:max_keypoints])
                
# #                 # Keypoints 시각적으로 프레임에 그리기
# #                 for i in range(0, len(normalized_keypoints), 2):
# #                     x, y = int(normalized_keypoints[i] * frame_width), int(normalized_keypoints[i + 1] * frame_height)
# #                     cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 녹색 원으로 keypoint 그리기
                
# #         # 화면에 프레임 보여주기
# #         cv2.imshow('Keypoints on Video', frame)
        
# #         # 'q' 키를 누르면 종료
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break
    
# #     cap.release()
# #     cv2.destroyAllWindows()

# #     # keypoints_sequence를 배열로 변환
# #     keypoints_sequence = np.array(keypoints_sequence)

# #     return keypoints_sequence

# # # 두 시퀀스 간의 DTW 거리 계산
# # def calculate_dtw_distance(seq1, seq2):
# #     # 확인: 두 시퀀스의 모양
# #     print(f"Shape of seq1: {seq1.shape}")
# #     print(f"Shape of seq2: {seq2.shape}")

# #     # 각 시퀀스를 2D로 변환 (프레임 개수는 다를 수 있음)
# #     min_len = min(len(seq1), len(seq2))
# #     seq1_flat = seq1[:min_len]  # 두 시퀀스의 길이를 동일하게 맞춤
# #     seq2_flat = seq2[:min_len]

# #     # 프레임별로 DTW 거리 계산 (1차원 배열로 처리)
# #     distances = []
# #     for i in range(min_len):
# #         distance = dtw.distance(seq1_flat[i], seq2_flat[i])
# #         distances.append(distance)

# #     # 전체 평균 DTW 거리 반환
# #     return np.mean(distances)

# # # 두 영상의 유사도를 계산하는 메인 함수
# # def compare_videos(video_path1, video_path2, model):
# #     # 첫 번째 비디오에서 keypoints 시퀀스를 추출하고 화면에 표시
# #     keypoints_seq1 = extract_keypoints_and_display_video(video_path1, model)
# #     # 두 번째 비디오에서 keypoints 시퀀스를 추출하고 화면에 표시
# #     keypoints_seq2 = extract_keypoints_and_display_video(video_path2, model)

# #     # 두 시퀀스 간의 DTW 거리 계산
# #     dtw_distance = calculate_dtw_distance(keypoints_seq1, keypoints_seq2)
    
# #     # 유사도를 계산 (거리가 작을수록 더 유사함)
# #     print(f"DTW Distance between the two videos: {dtw_distance}")

# # # 예시 실행
# # video1_path = 'datasets/video6.mp4'  # 첫 번째 영상 경로
# # video2_path = 'datasets/video1.mp4'  # 두 번째 영상 경로

# # compare_videos(video1_path, video2_path, model)

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from dtaidistance import dtw

# # YOLO 모델 불러오기
# model = YOLO('yolov8n-pose.pt')  # YOLOv8 포즈 모델 경로

# # keypoints 좌표를 [0, 1]로 정규화하는 함수
# def normalize_keypoints(keypoints, frame_width, frame_height):
#     # x 좌표는 프레임 너비로, y 좌표는 프레임 높이로 나눠서 정규화
#     normalized_keypoints = np.copy(keypoints)
#     for i in range(0, len(keypoints), 2):
#         normalized_keypoints[i] = keypoints[i] / frame_width  # x 좌표
#         normalized_keypoints[i + 1] = keypoints[i + 1] / frame_height  # y 좌표
#     return normalized_keypoints

# # Keypoints 시퀀스를 스무딩하는 함수
# def smooth_keypoints(sequence, window_size=3):
#     smoothed_sequence = []
#     for i in range(sequence.shape[1]):  # 각 keypoint에 대해
#         smoothed = np.convolve(sequence[:, i], np.ones(window_size)/window_size, mode='valid')
#         smoothed_sequence.append(smoothed)
#     smoothed_sequence = np.array(smoothed_sequence).T
#     return smoothed_sequence

# # 비디오에서 keypoints 추출하고 시각적으로 보여주는 함수
# def extract_keypoints_and_display_video(video_path, model):
#     cap = cv2.VideoCapture(video_path)
#     keypoints_sequence = []
#     max_keypoints = 34  # Keypoints 배열의 고정된 크기 (17개의 keypoints, 각 2D 좌표)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # YOLO로 프레임에서 포즈 추출
#         results = model(frame)

#         for result in results:
#             if result.keypoints is not None:
#                 # Keypoints 추출 (xy 좌표만 사용)
#                 keypoints = result.keypoints.xy.cpu().numpy()  # NumPy 배열로 변환
#                 xy_keypoints = keypoints.flatten()  # 1D로 평탄화
                
#                 # 좌표 정규화
#                 normalized_keypoints = normalize_keypoints(xy_keypoints, frame_width, frame_height)

#                 # Keypoints 배열의 크기를 고정 (34로 맞춤, 부족하면 0으로 패딩)
#                 if len(normalized_keypoints) < max_keypoints:
#                     padded_keypoints = np.zeros(max_keypoints)
#                     padded_keypoints[:len(normalized_keypoints)] = normalized_keypoints
#                     keypoints_sequence.append(padded_keypoints)
#                 else:
#                     keypoints_sequence.append(normalized_keypoints[:max_keypoints])
                
#                 # Keypoints 시각적으로 프레임에 그리기
#                 for i in range(0, len(normalized_keypoints), 2):
#                     x, y = int(normalized_keypoints[i] * frame_width), int(normalized_keypoints[i + 1] * frame_height)
#                     cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 녹색 원으로 keypoint 그리기
                
#         # 화면에 프레임 보여주기
#         cv2.imshow('Keypoints on Video', frame)
        
#         # 'q' 키를 누르면 종료
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

#     # keypoints_sequence를 배열로 변환
#     keypoints_sequence = np.array(keypoints_sequence)

#     # keypoints 시퀀스에 스무딩 적용
#     if len(keypoints_sequence) > 3:  # 스무딩 적용 가능한 최소 길이 확인
#         keypoints_sequence = smooth_keypoints(keypoints_sequence)

#     return keypoints_sequence

# # 두 시퀀스 간의 DTW 거리 계산
# def calculate_dtw_distance(seq1, seq2):
#     # 확인: 두 시퀀스의 모양
#     print(f"Shape of seq1: {seq1.shape}")
#     print(f"Shape of seq2: {seq2.shape}")

#     # 각 시퀀스를 2D로 변환 (프레임 개수는 다를 수 있음)
#     min_len = min(len(seq1), len(seq2))
#     seq1_flat = seq1[:min_len]  # 두 시퀀스의 길이를 동일하게 맞춤
#     seq2_flat = seq2[:min_len]

#     # 프레임별로 DTW 거리 계산 (1차원 배열로 처리)
#     distances = []
#     for i in range(min_len):
#         distance = dtw.distance(seq1_flat[i], seq2_flat[i])
#         distances.append(distance)

#     # 전체 평균 DTW 거리 반환
#     return np.mean(distances)

# # 두 영상의 유사도를 계산하는 메인 함수
# def compare_videos(video_path1, video_path2, model):
#     # 첫 번째 비디오에서 keypoints 시퀀스를 추출하고 화면에 표시
#     keypoints_seq1 = extract_keypoints_and_display_video(video_path1, model)
#     # 두 번째 비디오에서 keypoints 시퀀스를 추출하고 화면에 표시
#     keypoints_seq2 = extract_keypoints_and_display_video(video_path2, model)

#     # 두 시퀀스 간의 DTW 거리 계산
#     dtw_distance = calculate_dtw_distance(keypoints_seq1, keypoints_seq2)
    
#     # 유사도를 계산 (거리가 작을수록 더 유사함)
#     print(f"DTW Distance between the two videos: {dtw_distance}")

# # 예시 실행
# video1_path = 'datasets/video6.mp4'  # 첫 번째 영상 경로
# video2_path = 'datasets/video61.mp4'  # 두 번째 영상 경로

# compare_videos(video1_path, video2_path, model)


import cv2
import numpy as np
from ultralytics import YOLO
from dtaidistance import dtw

# YOLO 모델 불러오기
model = YOLO('yolov8n-pose.pt')  # YOLOv8 포즈 모델 경로

# keypoints 좌표를 [0, 1]로 정규화하는 함수
def normalize_keypoints(keypoints, frame_width, frame_height):
    normalized_keypoints = np.copy(keypoints)
    for i in range(0, len(keypoints), 2):
        normalized_keypoints[i] = keypoints[i] / frame_width  # x 좌표
        normalized_keypoints[i + 1] = keypoints[i + 1] / frame_height  # y 좌표
    return normalized_keypoints

# Keypoints 간 상대적 거리 계산
def calculate_relative_distances(keypoints):
    num_keypoints = len(keypoints) // 2
    relative_distances = []
    
    # 각 keypoint 사이의 거리를 계산 (유클리드 거리)
    for i in range(num_keypoints):
        for j in range(i + 1, num_keypoints):
            x1, y1 = keypoints[2 * i], keypoints[2 * i + 1]
            x2, y2 = keypoints[2 * j], keypoints[2 * j + 1]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            relative_distances.append(distance)
    
    return np.array(relative_distances)

# Keypoints 시퀀스를 스무딩하는 함수
def smooth_keypoints(sequence, window_size=3):
    smoothed_sequence = []
    for i in range(sequence.shape[1]):  # 각 keypoint에 대해
        smoothed = np.convolve(sequence[:, i], np.ones(window_size)/window_size, mode='valid')
        smoothed_sequence.append(smoothed)
    smoothed_sequence = np.array(smoothed_sequence).T
    return smoothed_sequence

# 비디오에서 keypoints 추출하고 시각적으로 보여주는 함수
def extract_keypoints_and_display_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []
    max_keypoints = 34  # Keypoints 배열의 고정된 크기 (17개의 keypoints, 각 2D 좌표)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO로 프레임에서 포즈 추출
        results = model(frame)

        for result in results:
            if result.keypoints is not None:
                # Keypoints 추출 (xy 좌표만 사용)
                keypoints = result.keypoints.xy.cpu().numpy()  # NumPy 배열로 변환
                xy_keypoints = keypoints.flatten()  # 1D로 평탄화
                
                # 좌표 정규화
                normalized_keypoints = normalize_keypoints(xy_keypoints, frame_width, frame_height)

                # Keypoints 배열의 크기를 고정 (34로 맞춤, 부족하면 0으로 패딩)
                if len(normalized_keypoints) < max_keypoints:
                    padded_keypoints = np.zeros(max_keypoints)
                    padded_keypoints[:len(normalized_keypoints)] = normalized_keypoints
                    keypoints_sequence.append(padded_keypoints)
                else:
                    keypoints_sequence.append(normalized_keypoints[:max_keypoints])
                
                # Keypoints 시각적으로 프레임에 그리기
                for i in range(0, len(normalized_keypoints), 2):
                    x, y = int(normalized_keypoints[i] * frame_width), int(normalized_keypoints[i + 1] * frame_height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 녹색 원으로 keypoint 그리기
        
        # 동적으로 윈도우 크기 설정
        cv2.namedWindow('Keypoints on Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Keypoints on Video', 800, 600)
        
        # 화면에 프레임 보여주기
        cv2.imshow('Keypoints on Video', frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    # keypoints_sequence를 배열로 변환
    keypoints_sequence = np.array(keypoints_sequence)

    # keypoints 시퀀스에 스무딩 적용
    if len(keypoints_sequence) > 3:  # 스무딩 적용 가능한 최소 길이 확인
        keypoints_sequence = smooth_keypoints(keypoints_sequence)

    return keypoints_sequence

# 두 시퀀스 간의 DTW 거리 계산 (상대적 거리 기반)
def calculate_dtw_distance(seq1, seq2):
    # 각 시퀀스의 상대적 거리 계산
    seq1_relative = np.array([calculate_relative_distances(frame) for frame in seq1])
    seq2_relative = np.array([calculate_relative_distances(frame) for frame in seq2])

    # 시퀀스 길이 맞추기
    min_len = min(len(seq1_relative), len(seq2_relative))
    seq1_flat = seq1_relative[:min_len]  # 두 시퀀스의 길이를 동일하게 맞춤
    seq2_flat = seq2_relative[:min_len]

    # 프레임별로 DTW 거리 계산
    distances = []
    for i in range(min_len):
        distance = dtw.distance(seq1_flat[i], seq2_flat[i])
        distances.append(distance)

    return np.mean(distances)

# 두 영상의 유사도를 계산하는 메인 함수
def compare_videos(video_path1, video_path2, model):
    # 첫 번째 비디오에서 keypoints 시퀀스를 추출하고 화면에 표시
    keypoints_seq1 = extract_keypoints_and_display_video(video_path1, model)
    # 두 번째 비디오에서 keypoints 시퀀스를 추출하고 화면에 표시
    keypoints_seq2 = extract_keypoints_and_display_video(video_path2, model)

    # 두 시퀀스 간의 DTW 거리 계산 (상대적 거리 기반)
    dtw_distance = calculate_dtw_distance(keypoints_seq1, keypoints_seq2)
    
    # 유사도를 계산 (거리가 작을수록 더 유사함)
    print(f"DTW Distance between the two videos: {dtw_distance}")

# 예시 실행
video1_path = 'datasets/video6.mp4'  # 첫 번째 영상 경로
video2_path = 'datasets/video6111.mp4'  # 두 번째 영상 경로

compare_videos(video1_path, video2_path, model)
