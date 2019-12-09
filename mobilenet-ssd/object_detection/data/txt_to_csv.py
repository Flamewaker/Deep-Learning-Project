# -------------------------------------------------------------------------------------------------------
# 将txt文件转为csv文件
# -------------------------------------------------------------------------------------------------------

import pandas as pd
import cv2

d = ['person']
l = []
f = open('./person_test.txt', 'r')
xml_df = None
for lines in f:
    out = lines.split(" ")
    print(out)

    length = len(out)
    name = out[0].split("/")[-1]
    image = cv2.imread(out[0])
    print(out[0])
    (h, w) = image.shape[:2]

    for i in range(1, length):
        location = out[i].split(",")

        xmin = location[0]
        ymin = location[1]
        xmax = location[2]
        ymax = location[3]
        num = d[int(location[4])]
        value = (name, w, h, num, xmin, ymin, xmax, ymax)
        l.append(value)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(l, columns=column_name)

xml_df.to_csv('./person_test.csv', index=None)