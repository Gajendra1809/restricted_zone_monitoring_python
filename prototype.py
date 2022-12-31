#https://github.com/AlexanderMelde/SPHAR-Dataset/tree/master/videos
#cctv dataset
import cv2
import numpy as np
from time import sleep

larmin=20 
altmin=20 

offset=6  

poslin=150 

delay= 60 

detec = []
carros= 0

	
def page_cen(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('person detected.mp4')
#subt = cv2.bgsegm.createBackgroundSubtractorMOG()
subt=cv2.createBackgroundSubtractorMOG2()
ab=0
while True:
    ret , frame1 = cap.read()
    tempo = float(1/delay)
    sleep(tempo) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subt.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilate = cv2.morphologyEx (dilate, cv2. MORPH_CLOSE , kernel)
    contor,h=cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (20, poslin), (150, poslin), (255,127,0), 3) 

    for(i,c) in enumerate(contor):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contor = (w >= larmin) and (h >= altmin)
        if not validar_contor:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        centro = page_cen(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0,255), -1)

        if(ab==1):
            cv2.line(frame1, (20, poslin), (150, poslin), (0,127,255), 3)
            continue
        
        for (x,y) in detec:
            if y<(poslin+offset) and y>(poslin-offset):
                
                carros+=1
                #cv2.line(frame1, (25, poslin), (1200, poslin), (0,127,255), 3) 
                cv2.line(frame1, (20, poslin), (150, poslin), (255,127,0), 3) 
                detec.remove((x,y))
                print("Person Detected ")
                ab=1
                break
        if(ab==1):
            break     
                      
       
    #cv2.putText(frame1, "VEHICLE COUNT : "+str(carros), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    #cv2.imshow("Detectar",dilate)

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()