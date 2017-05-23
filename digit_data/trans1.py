import cv2
import os
import os.path
import numpy as np

# rootdir = "E:/githubLib/tensorTest/digit_data/trainingDigits"
s_dir = "digit_data/testDigits/"
d_dir = "digit_data/testImages/"
def eachFile(filepath):
    count = 0
    labels = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        print(allDir)
        if allDir.endswith('.txt'):
            inputf = open(os.path.join(s_dir, allDir), 'r')
            imgData = [[None] * 32 for i in range(32)]
            for i in range(32):
                for j in range(32):
                    imgData[i][j] = int(line[i][j])
            img = np.zeros((32, 32, 1), np.uint8)

            for i in range(32):
                for j in range(32):
                    img[i,j] = imgData[i][j] * 255

            img_outname = d_dir + "_" + str(count) + ".bmp"

            for label in range(10):
                if allDir.startswith(str(label)):
                    pair = (label, img_outname)
                    labels.append(pair)

            cv2.imwrite(img_outname, img)
            count = count + 1
    output = open(os.path.join(d_dir, "labels.txt"), 'w')
    output.writelines(str(labels))
