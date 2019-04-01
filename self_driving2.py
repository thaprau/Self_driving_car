import cv2
import numpy as np
import matplotlib.pyplot as plt

video = True

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(0, height),(190, 0), (280, 0), (530, height)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny



cap = cv2.VideoCapture("test1.mp4")
frame = cv2.imread('Test_pic.jpeg')
IMAGE_H = 150
IMAGE_W = 600

src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
#src = np.float32([[0, IMAGE_H], [510, IMAGE_H], [200, 0], [300, 0]])
#dst = np.float32([[200, IMAGE_H], [300, IMAGE_H], [0, 0], [IMAGE_W, 0]])
dst = np.float32([[202, IMAGE_H], [265, IMAGE_H], [0, 0], [IMAGE_W, 0]])
M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

if(video):
    while(cap.isOpened):
        _, frame = cap.read()

        #Modify image
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #canny_image = cv2.Canny(gray, 100, 200)
        frame = frame[350:(350+IMAGE_H), 400:1000] # Apply np slicing for ROI crop'
        warped = cv2.warpPerspective(frame, M, (IMAGE_W, IMAGE_H))


        
        cv2.imshow('reslut', warped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

else:
    
    

    


    frame = frame[350:(350+IMAGE_H), 400:1000] # Apply np slicing for ROI crop'
    
    canny_image = canny(frame)
    #masked = region_of_interest(canny_image)
    warped_image = cv2.warpPerspective(canny_image, M, (IMAGE_W, IMAGE_H))
    warped_image_real = cv2.warpPerspective(frame, M, (IMAGE_W, IMAGE_H))
    #lines = cv2.HoughLinesP(warped_image, 2, np.pi/180, 100, np.array([]), minLineLength=20, maxLineGap=25)
    #lines_image = display_lines(warped_image, lines)
    #lines_normal = cv2.warpPerspective(warped_image, Minv, (IMAGE_W, IMAGE_H)) 

    plt.imshow(warped_image)
    plt.show()
