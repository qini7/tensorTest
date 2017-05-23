import cv2
import os
import os.path
import numpy as np

# rootdir = "E:/githubLib/tensorTest/digit_data/trainingDigits"
#s_dir = "digit_data/testDigits/"
#d_dir = "digit_data/testImages/"
s_dir = "digit_data/trainingDigits/"
d_dir = "digit_data/trainImages/"

def eachFile(filepath):
    count = 0
    labels = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        #print(allDir)
        if allDir.endswith('.txt'):
            inputf = open(os.path.join(s_dir, allDir), 'r')
            #warning: if use [[None] * 4] * 4, the 4 vector will duplicate each other
            #in some forum, this case is called "low copy"
            imgData = [[None] * 32 for i in range(32)]
            line = inputf.readlines()
            #print(line)
            for i in range(32):
                for j in range(32):
                    #print(i, j)
                    imgData[i][j] = int(line[i][j])
                    #print(imgData)

            #for i,j in range(32, 32):
            #    imgData[i, j] = int(line[i, j])
            #print(imgData)
            #transform imgData to img
            img = np.zeros((32, 32, 1), np.uint8)
            for i in range(32):
                for j in range(32):
                    img[i,j] = imgData[i][j] * 255
            #print(img)
            #cv2.imshow("img", img)
            #cv2.waitKey(0)

            img_outname = d_dir + "_" + str(count) + ".bmp"

            for label in range(10):
                if allDir.startswith(str(label)):
                    pair = (label, img_outname)
                    labels.append(pair)

            cv2.imwrite(img_outname, img)
            count = count + 1

    output = open(os.path.join(d_dir, "labels.txt"), 'w')
    #output.write(str(labels))
    for label in labels:
        #print(label[0], label[1])
        output.write(str(label[0]) + " ")
        output.write(str(label[1]) + "\n")
eachFile(s_dir)
