import cv2
 
# Opens the Video file
cap= cv2.VideoCapture('WhatsApp Video 2019-12-01 at 12.28.46 PM.mp4')
i=0
while(cap.isOpened()):
    
    ret, frame = cap.read()
    #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if ret == False:
        break
    cv2.imwrite('orig_frames/frame'+str(i)+'.jpg',frame)
    img = cv2.imread('orig_frames/frame'+str(i)+'.jpg')
    img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # img_rotate_90_clockwise = cv2.rotate(img_rotate_90_clockwise, cv2.ROTATE_90_CLOCKWISE)
    # img_rotate_90_clockwise = cv2.rotate(img_rotate_90_clockwise, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite('orig_rotated_frames/frame'+str(i)+'.jpg', img_rotate_90_clockwise)
	
    i+=1

 
cap.release()
cv2.destroyAllWindows()