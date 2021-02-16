## this file you can use in the server where all the folder with the runs are. It creates a folder with all the summaries

from glob import glob


for filepath in glob("./*/quantifiers.csv"):
    new_filename_base = filepath.split("/")[1]
    with open(filepath, "rb") as openfile:
        content = openfile.read()

    with open("./summaries/" + new_filename_base + ".csv", "wb") as openfile:
        openfile.write(content)