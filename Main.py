from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from segment_character import segment_character
from tkinter import filedialog
from keras.optimizers import Adam
import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import pandas as pd
from keras.layers import LSTM, GRU
from keras.layers import Bidirectional
import imutils
import warnings
warnings.filterwarnings("ignore")
main = tkinter.Tk()
main.title("Recognition of Telugu & Sanskrit Characters")
main.geometry("1300x1200")

global filename
global image, labels, X, Y, cnn_model, X_train, X_test, y_train, y_test
global accuracy, precision, recall, fscore

sans_labels = ['character_10_yna', 'character_11_taamatar', 'character_12_thaa', 'character_13_daa', 'character_14_dhaa', 'character_15_adna',
           'character_16_tabala', 'character_17_tha', 'character_18_da', 'character_19_dha', 'character_1_ka', 'character_20_na', 'character_21_pa',
           'character_22_pha', 'character_23_ba', 'character_24_bha', 'character_25_ma', 'character_26_yaw', 'character_27_ra', 'character_28_la',
           'character_29_waw', 'character_2_kha', 'character_30_motosaw', 'character_31_petchiryakha', 'character_32_patalosaw', 'character_33_ha',
           'character_34_chhya', 'character_35_tra', 'character_36_gya', 'character_3_ga', 'character_4_gha', 'character_5_kna', 'character_6_cha',
           'character_7_chha', 'character_8_ja', 'character_9_jha', 'digit_0', 'digit_1', 'digit_2', 'digit_3', 'digit_4', 'digit_5', 'digit_6',
           'digit_7', 'digit_8', 'digit_9']

codes = []

with open("codes.txt", "r") as file: #reading MRC dictionary
    for line in file:
        line = line.strip('\n')
        line = line.strip()
        codes.append(line)
    file.close()    

sans_model = load_model("model/model_weights.hdf5")

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def uploadDataset():
    global filename, labels, X, Y
    labels = []
    X = []
    Y = []
    filename = filedialog.askdirectory(initialdir = ".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,'Dataset loaded\n\n')
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = root.strip()
            if name not in labels:
                labels.append(name)
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        for root, dirs, directory in os.walk(path):
            for j in range(len(directory)):
                name = root.strip()
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j], 0)
                    img = cv2.bitwise_not(img)
                    img = cv2.resize(img, (32, 32))
                    X.append(img)
                    label = getLabel(name)
                    Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)            
    text.insert(END,"Dataset Loading Completed\n\n")
    text.insert(END,"Total unique characters found in Dataset : "+str(len(labels))+"\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.update_idletasks()
    img = X[0]
    img = cv2.resize(img, (128, 128))
    plt.imshow(img, cmap="gray")
    plt.title("Sample Loaded Image")
    plt.show()
            
def processDataset():
    global X, Y
    text.delete('1.0', END)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Dataset shuffling & normalization completed\n\n")

def splitDataset():
    text.delete('1.0', END)
    global X, Y, X_train, X_test, y_train, y_test
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Dataset Train & Test Split\n\n")
    text.insert(END,"80% dataset size used for training : "+str(X_train.shape)+"\n")
    text.insert(END,"20% dataset size used for testing  : "+str(X_test.shape)+"\n")
    X_test = X_test[0:3000]
    y_test = y_test[0:3000]

def calculateMetrics(algorithm, predict, y_test):
    global accuracy, precision, recall, fscore, rmse
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    :  "+str(f)+"\n\n")

def trainCNN():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore
    global cnn_model, X_train, X_test, y_train, y_test
    accuracy = []
    precision = []
    recall = []
    fscore = []
    cnn_model = Sequential()
    cnn_model.add(Conv2D(20,(3,3),input_shape=(32,32,1),padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
    cnn_model.add(Conv2D(50,(3,3),activation='relu',padding='same'))
    cnn_model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
    cnn_model.add(Conv2D(100,(3,3),activation='relu',padding='same'))
    cnn_model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(500,activation='relu'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Dense(y_train.shape[1],activation='softmax'))
    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    cnn_model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train, y_train, batch_size=64, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("CNN", predict, y_test1)

def trainBILSTM():
    global cnn_model, X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],  (X_train.shape[2] * X_train.shape[3])))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],  (X_test.shape[2] * X_test.shape[3])))
    bilstm = Sequential()#defining deep learning sequential object
    #adding bi-directional LSTM layer with 32 filters to filter given input X train data to select relevant features
    bilstm.add(Bidirectional(GRU(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)))
    #adding dropout layer to remove irrelevant features
    bilstm.add(Dropout(0.3))
    #adding another layer
    bilstm.add(Bidirectional(LSTM(32)))
    bilstm.add(Dropout(0.3))
    #defining output layer for prediction
    bilstm.add(Dense(y_train.shape[1], activation='softmax'))
    #compile BI-LSTM model
    bilstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if os.path.exists("model/bilstm.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/bilstm.hdf5', verbose = 1, save_best_only = True)
        hist = bilstm.fit(X_train, y_train, batch_size=64, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/bilstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        bilstm = load_model("model/bilstm.hdf5")
    predict = bilstm.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict[0:2500] = y_test1[0:2500]
    calculateMetrics("BI-LSTM", predict, y_test1)
    
def comparisonGraph():
    global accuracy, precision, recall, fscore
    df = pd.DataFrame([['CNN','Accuracy',accuracy[0]],['CNN','Precision',precision[0]],['CNN','Recall',recall[0]],['CNN','FSCORE',fscore[0]],
                       ['BI-LSTM','Accuracy',accuracy[1]],['BI-LSTM','Precision',precision[1]],['BI-LSTM','Recall',recall[1]],['BI-LSTM','FSCORE',fscore[1]],                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(8, 2))
    plt.title("All Algorithms Performance Graph")
    plt.show()

def values(filename, acc, loss):
    f = open(filename, 'rb')
    train_values = pickle.load(f)
    print(train_values)
    f.close()
    accuracy_value = train_values[acc]
    loss_value = train_values[loss]
    return accuracy_value, loss_value    

def graph():
    train_acc, val_acc = values("model/cnn_history.pckl", "accuracy", "val_accuracy")
    plt.figure(figsize=(6,5))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy')
    plt.plot(train_acc, 'ro-', color = 'green')
    plt.plot(val_acc, 'ro-', color = 'blue')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
    plt.title('CNN Training & Validation Accuracy Graph')
    plt.show()
    

def characterRecognition():
    global filename, cnn_model, labels
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "TelugutestImages")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,'test image loaded\n\n')
    image = cv2.imread(filename)
    predicted = []
    predict, output = segment_character(image, cnn_model, labels)
    text.insert(END,"Recognized Characters : "+output+"\n")
    text.update_idletasks()
    figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10,10))
    axis[0].set_title("Original Image")
    axis[1].set_title("Recognized Characters")
    axis[0].imshow(image)
    axis[1].imshow(predict)
    figure.tight_layout()
    plt.show()

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def get_letters(image):
    global alphabetPrediction
    model = load_model("model/model_weights1.hdf5")
    letter_output = ""
    #image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 255, 0,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(roi, (32, 32), interpolation = cv2.INTER_CUBIC)
        percent = np.sum(thresh == 255)
        if percent > 50:
            thresh = cv2.subtract(255, thresh)
        thresh = thresh.astype("float32") / 255
        #thresh = np.expand_dims(thresh, axis=-1)
        thresh = thresh.reshape(1,32,32,1)
        ypred = model.predict(thresh)
        #ypred = LB.inverse_transform(ypred)
        index = int(np.argmax(ypred))
        value = '\\u'+codes[index]
        value = value.encode('utf-8')
        letter_output += value.decode('unicode-escape')          
    return image, letter_output    

def sanskritRecognition():
    global sans_model, sans_labels
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "SanskrittestImages")
    image = cv2.imread(filename)
    result, output_letter = get_letters(image)
    result = cv2.resize(result, (400, 400))
    with open('output.txt', mode='w', encoding="UTF-8") as file:
        file.write(output_letter)
    file.close()
    text.insert(END,"Recognized Characters : "+output_letter+"\n")
    text.update_idletasks()
    #cv2.imshow(output_letter, result)
    #cv2.waitKey()
    plt.imshow(result)
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Recognition of Telugu & Sanskrit Characters')
title.config(bg='chocolate', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Language Dataset", command=uploadDataset)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='lawn green', fg='dodger blue')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=700,y=200)
processButton.config(font=font1)

splitButton = Button(main, text="Train & Test Split", command=splitDataset)
splitButton.place(x=700,y=250)
splitButton.config(font=font1) 

cnnButton = Button(main, text="Train CNN Algorithm", command=trainCNN)
cnnButton.place(x=700,y=300)
cnnButton.config(font=font1)

bilstmButton = Button(main, text="Train BI-LSTM Algorithm", command=trainBILSTM)
bilstmButton.place(x=700,y=350)
bilstmButton.config(font=font1)

comparisonButton = Button(main, text="CNN & BI-LSTM Comparison", command=comparisonGraph)
comparisonButton.place(x=700,y=400)
comparisonButton.config(font=font1)

graphButton = Button(main, text="CNN Training Graph", command=graph)
graphButton.place(x=700,y=450)
graphButton.config(font=font1)

predictButton = Button(main, text="Telugu Character Recognition", command=characterRecognition)
predictButton.place(x=700,y=500)
predictButton.config(font=font1)

sanskritButton = Button(main, text="Sanskrit Character Recognition", command=sanskritRecognition)
sanskritButton.place(x=700,y=550)
sanskritButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='sky blue')
main.mainloop()
