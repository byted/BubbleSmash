import numpy as np
import sys
import cv2

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    im_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    skin_ycrcb_mint = np.array((0, 133, 77))
    skin_ycrcb_maxt = np.array((255, 173, 127))
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)

    # # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # faces = faceCascade.detectMultiScale(
    # gray,
    # scaleFactor=1.1,
    # minNeighbors=5,
    # minSize=(30, 30),
    # flags=0
    # )

    # for (x, y, w, h) in faces:
    #     cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', skin_ycrcb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
