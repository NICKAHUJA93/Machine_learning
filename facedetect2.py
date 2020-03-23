import cv2
import matplotlib.pyplot as plt

num_of_sample=200
vid=cv2.VideoCapture(0) #to open the camera
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

iter1=0
while(iter1<num_of_sample):
    r,frame=vid.read();
    frame=cv2.resize(frame,(640,480)) #resizing the frame
    im1=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) #gray scale conversion of color image
    face=face_cascade.detectMultiScale(im1)
    for x,y,w,h in(face):
        cv2.rectangle(frame,(x,y),(x+w,y+h),[255,0,0],4)
        iter1+=1
        im_f=im1[y:y+h,x:x+w]
        im_f=cv2.resize(im_f,(112,92))
        cv2.putText(frame,'face No. '+str(iter1),(x,y), cv2.FONT_ITALIC, 1,
                    (255,0,255),2,cv2.LINE_AA)
        path = 'C:/Users/sample/%d.png'%(iter1) #path to save the image
        cv2.imwrite(path,im_f)  #to save the image

cv2.imshow('frame',frame)
cv2.waitKey(1)
vid.release()
cv2.destroyAllWindows()

