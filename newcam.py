import datetime
from tensorflow.keras.models import load_model
#from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

#from keras.preprocessing.image import ImageDataGenerator

import numpy as np

#------------------------------
# sess = tf.Session()
# keras.backend.set_session(sess)
#------------------------------
#variables
from DBConnection import Db

num_classes =2
batch_size = 80
epochs = 10
#------------------------------
cam_id=6
import time

from tensorflow import keras
import os, cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.engine.saving import load_model
# manipulate with numpy,load with panda
import numpy as np

# data visualization
import cv2
model = Sequential()



def read_dataset(fpath):
    data_list = []
    img = cv2.imread(fpath,
                     cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
    data_list.append(res)
    return (np.asarray(data_list, dtype=np.float32))


fpath=r"E:\Pycharm_wamp_sql_adobe\Project\Violence_Detection\myapp\static\dataset\noFight\nofi001.mp4"
vs = cv2.VideoCapture(fpath)
cnt = 0
from pygame import mixer
mixer.init()
while True:
    ok, frame = vs.read()
    if ok:
        cv2.imwrite(r'E:\Pycharm_wamp_sql_adobe\Project\Violence_Detection\myapp\static\cap.jpg', frame)
        dataset=read_dataset(r'E:\Pycharm_wamp_sql_adobe\Project\Violence_Detection\myapp\static\cap.jpg')
        dataset = dataset / 255
        (mnist_row, mnist_col, mnist_color) = 48, 48, 1
        dataset = dataset.reshape(dataset.shape[0], mnist_row, mnist_col, mnist_color)
        #   violence
        mo = load_model(r"E:\Pycharm_wamp_sql_adobe\Project\Violence_Detection\model1.h5")
        yhat_classes = mo.predict_classes(dataset, verbose=0)
        print("Violence detection :  ", yhat_classes )
        if yhat_classes[0] == 0:    #   detected fight
            cnt+=1
        # if cnt == 10:
        #     print("Crossed fight threshold")
        #
        #     mixer.music.load(r'E:\Pycharm_wamp_sql_adobe\Project\Violence_Detection\myapp\static\alarm.mp3')
        #     mixer.music.play()
        #     time.sleep(3)
        #     d=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        #     cv2.imwrite(r"E:\Pycharm_wamp_sql_adobe\Project\Violence_Detection\myapp\static\pic\\" + d + ".jpg", frame)
        #     path="/static/pic/" + d + ".jpg"
        #     db=Db()
        #     db.insert("INSERT INTO `myapp_violence`(DATE, TIME, image, CAMERA_id) VALUES(CURDATE(), CURTIME(), '"+path+"', '"+str(cam_id)+"')")
        #     cnt=0

        #   weapon
        mo2 = load_model(r"E:\Pycharm_wamp_sql_adobe\Project\Violence_Detection\myapp\static\model1.h5")
        yhat_classes2 = mo2.predict_classes(dataset, verbose=0)
        print("Weapon detection   ", yhat_classes2 )
        if yhat_classes2[0] == 2 or yhat_classes2[0]==6:    #   detected weapon
            cnt+=1
        if cnt == 10:
            print("Crossed threshold")

            mixer.music.load(r'E:\Pycharm_wamp_sql_adobe\Project\Violence_Detection\myapp\static\alarm.mp3')
            mixer.music.play()
            time.sleep(3)
            d=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(r"E:\Pycharm_wamp_sql_adobe\Project\Violence_Detection\myapp\static\pic\\" + d + ".jpg", frame)
            path="/static/pic/" + d + ".jpg"
            db=Db()
            db.insert("INSERT INTO `myapp_violence`(DATE, TIME, image, CAMERA_id) VALUES(CURDATE(), CURTIME(), '"+path+"', '"+str(cam_id)+"')")
            cnt=0

        cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
