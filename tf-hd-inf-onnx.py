import numpy as np 
import pandas as pd
import cv2
import tensorflow as tf 
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import onnx
import onnxruntime


session = onnxruntime.InferenceSession('trained_models/20210820_134816_palm_clf1_efficientb2_96.onnx')


input_name = session.get_inputs()[0].name
print("input name", input_name)
input_shape = session.get_inputs()[0].shape
print("input shape", input_shape)
input_type = session.get_inputs()[0].type
print("input type", input_type)


IMG_SIZE = 96

CATEGORIES = ["Palm","Posterior"]

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) #0 

while True:

    suc, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dic = {"x": [], "y": []}
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for lndms in results.multi_hand_landmarks:
            for lm in (lndms.landmark):
                
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                dic["x"].append(cx)
                dic["y"].append(cy)

            df = pd.DataFrame(dic)
            x1, x2 = abs(df["x"].min()), abs(df["x"].max())
            y1, y2 = abs(df["y"].min()), abs(df["y"].max())
            
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),2)
            dic.update({"x": [], "y": []})
        

        crop = imgRGB[y1:y2,x1:x2]
        
        if crop.shape < (150,150):
            cv2.putText(img, "Hand is too far away!",(0,130), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)


        ##MODEL SIDE###
        
        image_np = cv2.resize(crop, (IMG_SIZE,IMG_SIZE))


        new_array = img_to_array(image_np)
        new_array = np.expand_dims(image_np, axis=0)
        new_array = np.array(new_array, dtype=np.float32)
        
        

        try:

            prediction = session.run([],{input_name:new_array})
            
        except:
            pass
            continue



        if prediction[0] > 0.5 :  #POSTERIOR
            classes = CATEGORIES[int(prediction[0].round())]
            cv2.putText(img, classes,(25,25), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
            cv2.putText(img, str(prediction[0]),(0,70), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
            cv2.putText(img, str(new_array.shape),(0,100), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,250),1)

        elif prediction[0] <= 0.5: #PALM 
            classes = CATEGORIES[int(prediction[0].round())]
            cv2.putText(img, classes,(25,25), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
            cv2.putText(img, str(prediction[0]),(0,70), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
            cv2.putText(img, str(new_array.shape),(0,100), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,250),1)
    else:
        cv2.putText(img, "No hands",(25,25), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)





    cv2.imshow("Inference", img)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()