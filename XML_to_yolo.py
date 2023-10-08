import xml.etree.ElementTree as ET
import glob
import os
import json


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


os.chdir("../../direct_sun/input")


def run_xml_to_yolo_or_standard(train=True, to_yolo=False):
    classes = []

    if train:
        input_dir = "ground-truth/"
        output_dir = "ground-truth/"
        image_dir = "images-optional/"
    else:
        input_dir = "yolov7_custom/pascal_val/phone/"
        output_dir = "data/val/phone/labels/"
        image_dir = "data/val/phone/images/"

    # create the labels folder (output directory)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # identify all the xml files in the annotations folder (input directory)
    files = glob.glob(os.path.join(input_dir, "*.xml"))
    print(input_dir)
    # loop through each
    for fil in files:
        basename = os.path.basename(fil)
        filename = os.path.splitext(basename)[0]
        # print(filename)    # check if the label contains the corresponding image file
        if not os.path.exists(os.path.join(image_dir, f"{filename}.jpg")):
            print(f"{filename} image does not exist!")
            continue

        result = []

        # parse the content of the xml file
        tree = ET.parse(fil)
        root = tree.getroot()
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)

        for obj in root.findall("object"):
            label = obj.find("name").text
            # print(label)
            # check for new classes and append to list
            if label not in classes:
                classes.append(label)
            index = classes.index(label)
            bbox = [int(x.text) for x in obj.find("bndbox")]
            print(label)
            if to_yolo:
                # Convert from standard to yolo
                bbox = xml_to_yolo_bbox(bbox, width, height)
                label = index
            # convert data to string
            bbox_string = " ".join([str(x) for x in bbox])
            result.append(f"{label} {bbox_string}")

        if result:
            # generate a YOLO format text file for each xml file
            with open(
                os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("\n".join(result))
    # generate the classes file as reference
    with open("classes.txt", "w", encoding="utf8") as f:
        f.write(json.dumps(classes))


run_xml_to_yolo_or_standard(train=True)
