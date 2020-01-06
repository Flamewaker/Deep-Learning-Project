import pandas as pd
import os
import csv

data = pd.read_csv('./voc_val.csv', encoding='utf-8')
with open('./voc_val.txt', 'a+', encoding='utf-8') as f:
    count = 0
    next_index = 0
    print(data.values)
    print(len(data.values))
    for index in range(len(data.values)):
        if index == next_index:
            p = next_index + 1
            while (data.values[index][0] == data.values[p][0]):
                if ( p == len(data.values) - 1): break
                p = p + 1
                count = count + 1
            print(count)
            if ( p == len(data.values) - 1):
                next_index = p + 1
            else:
                next_index = p
            count = count + 1
            for i in range(0, count):
                if (i == 0):
                    f.write((str(data.values[index][0]) + ' ' + str(data.values[index][1]) + ',' + str(
                        data.values[index][2]) + ',' + str(data.values[index][3]) + ',' + str(
                        data.values[index][4]) + ',' + str(data.values[index][5])))
                else:
                    f.write(' ' + str(data.values[index + i][1]) + ',' + str(data.values[index + i][2]) + ',' + str(
                        data.values[index + i][3]) + ',' + str(data.values[index + i][4]) + ',' + str(
                        data.values[index + i][5]))
            f.write("\n")
            count = 0


     # for line in data.values:
     #     f.write((str(line[0]) + ' ' + str(line[1]) + ',' + str(line[2]) + ',' + str(line[3]) + ',' + str(line[4]) + ',' + str(line[5])))
     #     f.write("\n")

print("successfully")
