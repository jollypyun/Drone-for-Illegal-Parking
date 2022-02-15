import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

MIN_AREA = 80
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 1.0

# 비디오 화면 창 출력
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

# esc 입력 시 탈출
while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 흑백화면 처리

    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # 직사각형의 구조화 요소 커널을 3x3 크기로 검출

    # 노이즈 제거
    imgTop = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect) # 밝은 부분 강조
    imgBlack = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect) # 어두운 부분 강조
    gray = cv2.subtract(cv2.add(gray, imgTop), imgBlack)

    imgBlur = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX=0) # 블러 처리
    imgThresh = cv2.adaptiveThreshold(imgBlur, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=19, C=9) # Threshold 처리

    contours, _ = cv2.findContours(imgThresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE) # Contour 처리
    temp_result = np.zeros((800, 800, 1), dtype=np.uint8)
    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
    temp_result = np.zeros((800, 800, 1), dtype=np.uint8)
    contours_dict = []
    cnt = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area, ratio = w*h, w/h
        if area > MIN_AREA and w > MIN_WIDTH and h > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
            idx = cnt
            cnt += 1
            cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
            contours_dict.append({'contour': contour, 'x': x, 'y': y, 'w': w, 'h': h, 'cx': x + (w/2), 'cy': y + (h/2), 'idx' : idx})

    for d1 in contours_dict:

        for d2 in contours_dict:
            if d1['idx'] == d2['idx']:
                continue
            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else: angle_diff = np.degrees(np.arctan(dy/dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['h']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']


    cv2.imshow("VideoFrame", temp_result)
capture.release()
cv2.destroyAllWindows()