import sys, os
import glob
import numpy as np
import skimage, skimage.filters
import pickle
import pandas as pd
import shutil
import json



for_eval = [
            "Cardiomegaly",
            "Mass",
            "Atelectasis",
            "Effusion",
            "Consolidation",
            "Edema"
            ]

page = ""
for target in for_eval:
    page += "<h2>{}</h2>".format(target)
    for label in [1, 0]:
        if label == 1:
            page += "<h3 style='margin-left: 20px;'>True Positives</h3>"
        else:
            page += "<h3 style='margin-left: 20px;'>False Positives</h3>"
        page += "<ol>"
        for j in sorted(glob.glob("images/*{}-{}.json".format(target, label))):
            metadata = json.load(open(j))
            print(metadata["id"])
            #title = "{} - {} - {}".format(metadata["source"]["Sex"], metadata["source"]["Age"], metadata["id"])
            title = "{} - {}".format(metadata["dataset"], metadata["id"])
            link = "<li><a href='{}' target='viewer'>{}</a></li>".format("viewer.htm?img="+metadata["id"], title)
            page += link
        page += "</ol>"
with open("list.htm", 'w') as f:
    f.write(page)
    
print("Done")

