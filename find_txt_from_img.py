import xml.etree.ElementTree as ET
import glob
import os
import json
import shutil


img_dir = "map-master-HE/input/images-optional/"
full_txt_dir = "map-master/input/ground-truth/"
new_txt_dir = "map-master-HE/input/ground-truth/"

# identify all the txt files from the subset of images
files = glob.glob(os.path.join(img_dir, "*.jpg"))

for fil in files:
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]
    shutil.copy(
        os.path.join(full_txt_dir, f"{filename}.txt"),
        os.path.join(new_txt_dir, f"{filename}.txt"),
    )

print("Files copies")
