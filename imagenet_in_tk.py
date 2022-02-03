import cv2
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk, ImageDraw, ImageFont
import time
import numpy as np

import tensorflow as tf
if int(tf.__version__.split('.')[0]) >= 2:
    from tensorflow import keras
else:
    import keras
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import json
import os

## 初期設定
# カメラの定義
cameraID = 0

# フォント
#fnt=font.Font(family="System",size=20,weight="bold")

# カレントディレクトリに移動
current_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_folder)

# 駆動中
isRunnning = True


## 識別
# ImageNetのラベル一覧を読み込む
with open('imagenet_class_index.json' , encoding='utf-8') as f:
    data = json.load(f)
    class_names = np.array([row['ja'] for row in data])

# VGG16 を構築する
model = VGG16()

# 分類
def classify(imgPIL):
    imgPIL = imgPIL.resize((224,224)) # resize
    x = image.img_to_array(imgPIL) # PIL -> np.float32
    x = np.expand_dims(x, axis=0) # (224,224,3) -> (1,224,224,3)
    x = preprocess_input(x) # 前処理

    # 推論する
    scores = model.predict(x)[0]
    top5_cls = scores.argsort()[-5:][::-1]

    # 推論結果を表示する
    lbl_answer.configure(text=class_names[top5_cls[0]])
    for i in range(5):
        lbl_prob[i].configure(text="確率{}位　　{:.2%}".format(i+1,scores[top5_cls[i]]))
        lbl_pred[i].configure(text="{}".format(class_names[top5_cls[i]]))


## 以下、tkinter

#カメラのサイズはたいてい4:3(640*480など)だが、VGG16が224*224の正方形で学習しているのでカメラサイズを設定する
camera_size = 480
cap = cv2.VideoCapture(cameraID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_size)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,camera_size)

# プログラム終了
def exit_app():
    cap.release()
    cv2.destroyAllWindows()
    root.quit()
    exit()

# テキストラベルをリセットする
def reset_lbl():
    lbl_answer.configure(text="")
    for i in range(5):
        lbl_pred[i].configure(text="")
        lbl_prob[i].configure(text="")

# 撮影時の演出
def effect(imgPIL):
    cx,cy = 240,240
    draw = ImageDraw.Draw(imgPIL)
    for i in [-1,1]:
        for j in [-1,1]:
            for x1,y1,x2,y2 in [(20,20,25,60),(20,20,60,25)]:
                x1 = cx+i*(cx-x1)+80
                y1 = cy+j*(cy-y1)
                x2 = cx+i*(cx-x2)+80
                y2 = cy+j*(cy-y2)
                draw.rectangle([(x1,y1),(x2,y2)], outline="white", fill="white")
    return imgPIL

# 撮影ボタンを押したときの動作
def pause_img():
    global imgTk
    global isRunnning

    # 駆動中ならば
    if isRunnning:
        isRunnning = False # 静止中にする
        ret, imgCV = cap.read() # カメラ取り込み
        imgPIL = cv2pil(imgCV) # PIL画像にする
        classify(imgPIL) # 推論する
        imgPIL = effect(imgPIL) # 演出
        imgTk = pil2tk(imgPIL) # 演出後の画像
        label_img.configure(image=imgTk)        
        btn.configure(image=imgBtnReset) # ボタン画像変更

    # 停止中ならば
    else:
        isRunnning=True # 駆動中にする
        btn.configure(image=imgBtnDo) # ボタン画像変更
        reset_lbl() # テキストラベルをリセットする

# メインループ
def main_loop():
    global isRunnning
    global imgTk

    # 駆動中ならば
    if isRunnning :
        ret, imgCV = cap.read() # カメラ取り込み
        imgTk = cv2tk(imgCV) # tk画像にする
        label_img.configure(image=imgTk) # ラベルに画像貼り付け

    k=cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        exit_app()

    root.after(1, main_loop) # 1ms後に自分自身を呼び出す

# 画像フォーマットを変換する
def cv2pil(imgCV):
    imgCV_RGB = cv2.cvtColor(imgCV, cv2.COLOR_BGR2RGB) # BGR -> RGB
    imgPIL = Image.fromarray(imgCV_RGB) # CV -> PIL
    return imgPIL

def cv2tk(imgCV):
    imgPIL = cv2pil(imgCV)
    imgTk = ImageTk.PhotoImage(imgPIL) # PIL -> ImageTk
    return imgTk

def pil2tk(imgPIL):
    imgTk = ImageTk.PhotoImage(imgPIL) # PIL -> ImageTk
    return imgTk


# labelのwidthやheightは文字が基準となっている。これをピクセルでサイズ指定する
def make_label(master, x, y, h, w, *args, **kwargs):
    f = tk.Frame(master, height=h, width=w)
    f.pack_propagate(0)
    f.place(x=x, y=y)
    label = tk.Label(f, *args, **kwargs)
    label.pack(fill=tk.BOTH, expand=1)
    return label

# tkinterのメイン
root = tk.Tk()
root.geometry("640x480")
root.title("VGG16 in tk")
#root.attributes('-fullscreen', True)

#フレーム（画面右に置くラベル群の親）
frame = tk.Frame(root, width=160, height=480)
frame.place(x=480, y=0)

# ボタン、ラベル
imgLogo = tk.PhotoImage(file="logo.png")
imgBtnDo = tk.PhotoImage(file="button_do.png")
imgBtnReset = tk.PhotoImage(file="button_reset.png")
lbl_logo = tk.Label(master=frame, image=imgLogo)
lbl_logo.place(x=0,y=0)
btn = tk.Button(master=frame, image=imgBtnDo, command=pause_img)
btn.place(x=0, y=320)

# 解答ラベル
lbl_answer=make_label(frame,0,61,60,160,font=("",16,"bold"),bg="lime")

# ラベルを配列変数として定義する
lbl_pred=[]
lbl_prob=[]
for i in range(5): # appendで追加定義されるので引数は不要
    lbl_prob.append(make_label(frame,0,120+40*i,20,160,anchor=tk.W))
    lbl_pred.append(make_label(frame,20,140+40*i,20,140,anchor=tk.W))

label_img=tk.Label(width=camera_size, height=camera_size)
label_img.place(x=0,y=0)

main_loop()

root.mainloop()
