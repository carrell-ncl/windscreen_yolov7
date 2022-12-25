import glob
import os
from PIL import Image


file_folder = r"data/val/phone/labels_old" # folder with .txt files to modify
img_folder = r"data/val/phone/images" # Image folder
output_label_folder = r"data/val/phone/labels" # Output folder for the modified .txt files

def xml_to_yolo_bbox(bbox:list, w:int, h:int):
    """Converts xmin, ymin, xmax, ymax to yolo

    Args:
        bbox (list): _description_
        w (int): Image width
        h (int): Image height


    """
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

# identify all the txt files in the annotations folder (input directory)
files = glob.glob(os.path.join(file_folder, '*.txt'))


# Make output directory to save yolo labels if it does not exist
if not os.path.exists(output_label_folder):
  os.makedirs(output_label_folder)

# loop through original .txt files
for fil in files:
  
  basename = os.path.basename(fil)
  filename = os.path.splitext(basename)[0]
  # check if the label contains the corresponding image file
  if not os.path.exists(os.path.join(img_folder, f"{filename}.jpg")):
      print(f"{filename} image does not exist!")
      continue

  # Obtain width and height of each image
  for image in os.listdir(img_folder):
      if os.path.isfile(os.path.join(img_folder, image)):
      name = basename.split('.txt')[0]
      if name in image:
        img = Image.open(f"{img_folder}/{image}")
        w, h = img.size

  if os.path.isfile(fil):
    # Stores each modified detection
    results = []
    with open(fil, "r") as file:
      data = file.readlines()
      for line in data:
        coords = line.split()[-4:] # Extract the original coordinates
        class_int = str(line.split()[0]) # Extract the class
        coords = [int(float(x)) for x in coords] # Convert to integer
        print(coords)

        yolo_coords = (xml_to_yolo_bbox(coords, w, h)) # Convert to Yolo

        yolo_coords = ' '.join([str(x) for x in yolo_coords]) # Convert detections to single string
        
        results.append(f"{class_int} {yolo_coords}") # Append results with class

    if results:
      with open(os.path.join(output_label_folder, basename), "w") as out_f:
        out_f.write("\n".join(results))
  



