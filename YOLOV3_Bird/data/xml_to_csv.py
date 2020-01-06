import xml.etree.ElementTree as ET
import pandas as pd
import glob

# -------------------------------------------------------------------------------------------------------


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            if member[0].text == 'bird':
                value = ("./data/images/" + root.find('filename').text,
                         int(member.find('bndbox')[0].text),
                         int(member.find('bndbox')[1].text),
                         int(member.find('bndbox')[2].text),
                         int(member.find('bndbox')[3].text),
                         member[0].text
                         )
                xml_list.append(value)
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

# -------------------------------------------------------------------------------------------------------


def main():
    xml_df = xml_to_csv("./annotations")
    xml_df.to_csv('./voc_train.csv', index=None)
    print('Successfully converted xml to csv.')


# -------------------------------------------------------------------------------------------------------

main()

# -------------------------------------------------------------------------------------------------------