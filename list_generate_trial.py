import sys, os
import glob
import numpy as np
import skimage, skimage.filters
import pickle
import pandas as pd
import shutil
import json
import random
import base64
import sklearn, sklearn.model_selection

for_eval = [
            "Cardiomegaly",
            "Atelectasis",
            "Effusion",
            "Lung Opacity",
            "Mass",
            "Pneumothorax"
            ]

number = 0
page = ""
page += """
<div style="width:100%;min-height:50px">
<img style="float:right;height:50px" src="assets/aimi-logo.png" />
<img style="float:right;height:50px" src="assets/stanford-logo2.png" />

<p style="max-width:700px;padding:5px">
The goal of a prediction explanation method is to identify features which are relevant to the prediction of a neural network and convey that information to the user.<br>
This trial is to determine if a new prediction explanation method performs better when explaining predictions on chest X-rays.<br>
Each link below corresponds to a single chest X-ray which has had a positive prediction made by a neural network. Method A corresponds to 3 traditional methods while method B corresponds to a new proposed method. More detailed explanations of the methods are available by clicking the "(What is this?)" link above an image.
</p>
</div>
"""
record = []
for target in for_eval:
    page += "<h2>{}</h2>".format(target)
    
    toprocess = sorted(glob.glob("images/*{}-*.json".format(target)))
    tp = [k.endswith("-1.json") for k in toprocess]
    
    groupa, groupb, groupa_l, groupb_l = sklearn.model_selection.train_test_split(toprocess, tp, stratify=tp, train_size=0.5, random_state=0)

    condition = ["Prediction Explanation Method A", "Prediction Explanation Method B"]
    
    
    
    for trial, group in enumerate([groupa, groupb]):
        
        page += "<h3 style='margin-left: 5px;'>{}</h3>".format(condition[trial])
        page += "<ul>"
        for j in group:
            number+=1
            print(j)
            metadata = json.load(open(j))
            metadata["trialid"] = number
            metadata["trial"] = trial
            record.append(metadata)
            print(metadata["id"])
            #title = "{} - {} - {}".format(metadata["source"]["Sex"], metadata["source"]["Age"], metadata["id"])
            title = "{} - (From {})".format(number, metadata["dataset"])
            imgb = base64.standard_b64encode(metadata["id"].encode('ascii')).decode("utf-8") 
            link = "<li><a href='viewer.htm?imgb={}&trial={}&id={}' target='viewer'>{}</a></li>".format(imgb, trial, number, title)
            page += link
        page += "</ul>"
with open("trial.htm", 'w') as f:
    f.write(page)
    
json.dump(record, open("trial_record.json", "w"))
    
print("Done")

