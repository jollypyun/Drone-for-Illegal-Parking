import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

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

    imgBlur = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX=0)
    imgThresh = cv2.adaptiveThreshold(imgBlur, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=19, C=9)

    contours, _ = cv2.findContours(imgThresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    temp_result = np.zeros((800,800, 1), dtype=np.uint8)


    cv2.imshow("VideoFrame", imgThresh)
capture.release()
cv2.destroyAllWindows()