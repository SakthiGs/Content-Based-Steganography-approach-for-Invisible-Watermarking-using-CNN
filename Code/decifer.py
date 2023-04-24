############################################################################################
################################## Working Code -- Testing #################################
############################################################################################

import cv2
import numpy as np
import argparse
import libsvm
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

def Decifer(img):
    Gap_Threshold = 55

    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    thresh = 255 - thresh

    key = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
           'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    rows_img = img.copy()
    boxes_img = img.copy()
    rowboxes = []
    rowcontours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rowcontours = rowcontours[0] if len(rowcontours) == 2 else rowcontours[1]
    index = 1

    for rowcntr in rowcontours:
        xr, yr, wr, hr = cv2.boundingRect(rowcntr)
        cv2.rectangle(rows_img, (xr, yr), (xr + wr, yr + hr), (0, 0, 255), 1)
        rowboxes.append((xr, yr, wr, hr))

    def takeSecond(elem):
        return elem[1]

    rowboxes.sort(key=takeSecond)

    nSize = 0
    nWidth = 0

    R = []
    model = keras.models.load_model('alpha.h5')
    for rowbox in rowboxes:
        # crop the image for a given row
        print(' ROW ', rowbox)
        xr = rowbox[0]
        yr = rowbox[1]
        wr = rowbox[2]
        hr = rowbox[3]
        row = thresh[yr:yr + hr, xr:xr + wr]
        bboxes = []
        bboxes_eliminate = []

        contours = cv2.findContours(row, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(contours)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            bboxes.append((x + xr, y + yr, w, h))

        def takeFirst(elem):
            return elem[0]

        bboxes.sort(key=takeFirst)
        # draw sorted boxes

        for box in bboxes:
            xb = box[0]
            yb = box[1]
            wb = box[2]
            hb = box[3]

            if wb * hb > 50:
                bboxes_eliminate.append((xb, yb, wb, hb))

        Out = ''
        temp = 0
        for i in bboxes_eliminate:

            nWidth += i[2]
            nSize = nSize + 1

            if (i[3] > i[2]):
                ROI = gray[i[1]:i[1] + i[3], i[0]:i[0] + i[2]]

                dim = (80, 80)
                ROI = cv2.resize(ROI, dim, interpolation=cv2.INTER_NEAREST)
                ROI = np.reshape(ROI, (1, 80, 80, 1))

                p_label = key[tf.argmax(model(ROI), axis=-1).numpy()[0]]

                for ii in range(1):
                    if ((i[0] - temp) > Gap_Threshold):
                        Out += ' '

                    Out += p_label

                index = index + 1
                temp = i[0]
            else:
                Out += ' -'

        R.append(Out)

    return R

    # show images
    # cv2.imshow("thresh", thresh)
    # cv2.imshow("morph", morph)
    # cv2.imshow("rows_img", rows_img)
    # cv2.imshow("boxes_img", boxes_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()