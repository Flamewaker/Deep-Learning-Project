import xml.etree.ElementTree as ET
import os

# ----------------------------------------------------------------------------------------------------------------------

l = os.listdir("./annotations/")
l.sort()
out = []
for i in range(0, len(l)):
    if l[i].endswith("xml"):
        out.append(l[i])

# ----------------------------------------------------------------------------------------------------------------------

for j in range(0, len(out)):
    root = ET.parse("./annotations/" + out[j]).getroot()
    object_num = len(root.getchildren())
    # print(object_num)

    count_1 = 0
    for k in range(0, object_num - 5):
        children_node = root.getchildren()[5 + k]
        print(list(children_node))
        print(len(list(children_node)))

        length = len(list(children_node))
        print(length)
        location = []
        for i in range(0, length):
            if i == 0:
                CLASS = list(children_node)[i].text.strip()

            if i == 4:
                for item in list(children_node)[i]:
                    location.append(item.text.strip())

                output = open("./train.txt", "a")

                if count_1 == 0:
                    output.write("./data/images/" + out[j].split(".")[0]
                                 + ".jpg" + " "
                                 + str(location[0]) + ","
                                 + str(location[1]) + ","
                                 + str(location[2]) + ","
                                 + str(location[3]) + ","
                                 + CLASS + " ")
                    count_1 += 1

                else:
                    output.write(str(location[0]) + ","
                                 + str(location[1]) + ","
                                 + str(location[2]) + ","
                                 + str(location[3]) + ","
                                 + CLASS + " ")

    output = open("./train.txt", "a")
    output.write("\n")


# ----------------------------------------------------------------------------------------------------------------------




