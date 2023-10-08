import os

file_folder = "data/val/phone/labels"

for file_name in os.listdir(file_folder):
    f = os.path.join(file_folder, file_name)
    print(f)

    with open(f, "r") as file:
        filedata = file.read()

    filedata = filedata.replace("Phone", "0")

    with open(f, "w") as file:
        file.write(filedata)
