import cv2
import numpy as np
import matplotlib.pyplot as plt

birds_view = False

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 100, 200)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(100, height),(1000, height), (640, 500)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(canny_image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image





if (1):
    cap = cv2.VideoCapture("test2.mp4")

    nr_of_iterations = 1
    iteration = 0
    if(birds_view):
        IMAGE_H = 223
        IMAGE_W = 1280
        src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
        dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
        M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
        Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation


    while(cap.isOpened()):
        _, frame = cap.read()
        if(birds_view):
            frame = frame[550:(550+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
            canny_image = canny(frame)
            warped_image = cv2.warpPerspective(canny_image, M, (IMAGE_W, IMAGE_H))
            lines = cv2.HoughLinesP(warped_image, 2, np.pi/180, 100, np.array([]), minLineLength=20, maxLineGap=25)
            #lines_image = display_lines(warped_image, lines)
            lines_normal = cv2.warpPerspective(warped_image, Minv, (IMAGE_W, IMAGE_H))
            cv2.imshow('result', lines_normal)

        else:
            canny_image = canny(frame)
            cropped_image = region_of_interest(canny_image)
            lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=20, maxLineGap=25)
            line_image = display_lines(frame, lines)
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
            #plt.imshow(combo_image)
            #plt.show()
            cv2.imshow('result', combo_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        iteration+=1
    cap.release()
    cv2.destroyAllWindows()





 