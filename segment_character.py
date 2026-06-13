import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

def getPredicted(image, cnn_model, labels):
    print(image.shape)
    img = image#cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, (32, 32))
    temp = []
    temp.append(img)
    img = np.asarray(temp)
    img = np.reshape(img, (img.shape[0], img.shape[1], img.shape[2], 1))
    img = img.astype('float32')
    img = img / 255
    predict = cnn_model.predict(img)
    predict = np.argmax(predict)
    predict = labels[predict]
    path = predict
    path = os.listdir(path)
    print(predict+"/"+path[0])
    img = cv2.imread(predict+"/"+path[0])
    img = cv2.resize(img, (64, 64))
    return img, os.path.basename(predict)

def segment_character(image, cnn_model, labels):
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    canny = cv2.Canny(blur, 50, 255, 1)
    im2, ctrs, hier = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (ctrs, boundingBoxes) = sort_contours(ctrs, method="left-to-right")
    predicted = []
    output = ""
    for i, cnt in enumerate(ctrs):
        x, y, w, h = cv2.boundingRect(cnt)
        if ((w*h) < 100):
            continue
        roi = gray[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        predict, chars = getPredicted(roi, cnn_model, labels)
        output += chars+" "
        predicted.append(predict)
        print(roi.shape)
        cv2.imwrite("output/"+str(i)+".png", roi)
    stacked_image = cv2.hconcat(predicted)    
    return stacked_image, output.strip()

    '''
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    min_area = 30
    image_number = 0
    predicted = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            roi = gray[y:y+h, x:x+w]
            predict = getPredicted(roi, cnn_model, labels)
            predicted.append(predict)
            plt.imshow(predict)
            plt.show()
    temp = []
    j = len(predicted) - 1
    while j >= 0:
        temp.append(predicted[j])
        j = j - 1
    stacked_image = cv2.hconcat(temp)    
    return stacked_image
    '''

'''
def segment_character(image, directory, main_image, cnn_model, labels):
    row, col = image.shape
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    im2, ctrs, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (ctrs, boundingBoxes) = sort_contours(ctrs, method="left-to-right")
    characters = {}
    ottaksharas = {}
    count = 0
    predicted = []
    for i, cnt in enumerate(ctrs):
        x, y, w, h = cv2.boundingRect(cnt)
        if ((w*h) < 100):
            continue
        roi = thresh2[y:y+h, x:x+w]
        cv2.rectangle(main_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        predict = getPredicted(roi, cnn_model, labels)
        predicted.append(predict)
        print(roi.shape)
        cv2.imwrite("output/"+str(i)+".png", roi)
    stacked_image = cv2.hconcat(predicted)    
    return stacked_image
'''        
