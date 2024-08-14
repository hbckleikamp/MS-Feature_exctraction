# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:07:48 2024

@author: hkleikamp
"""

import os
from pathlib import Path
from inspect import getsourcefile
os.chdir(str(Path(os.path.abspath(getsourcefile(lambda:0))).parents[0])) # set base path
basedir=os.getcwd()
print(basedir)

import subprocess

#%%

folders=[
"F:/Hugo/MNX-UD-neg-001.d",
"F:/Hugo/ACN-001.d",
"F:/Hugo/ACN-002.d",
"F:/Hugo/ACN-004.d",
"F:/Hugo/ACN-005.d",
"F:/Hugo/ACN-006.d",
"F:/Hugo/BM-001.d",
"F:/Hugo/DMIT-600pgul-001.d",
"F:/Hugo/DMIT-C18-neg-001.d",
"F:/Hugo/DMIT-neg-002.d",
"F:/Hugo/DMIT-neg-002_rainbow_test.d",
"F:/Hugo/DMIT-pos-001.d",
"F:/Hugo/MNT-2TBA-001.d",
"F:/Hugo/MNT-2TBA-C18-neg-001.d",
"F:/Hugo/MNT-TBA-001.d",
"F:/Hugo/MNT-TBA-pos-001.d",
"F:/Hugo/MNX-DMSO-neg-001.d",
"F:/Hugo/MNX-DMSO-neg-1250-001.d",
"F:/Hugo/MNX-DMSO-pos-1250-001.d",
"F:/Hugo/MNX-MeOH-neg-001.d",
"F:/Hugo/MNX-MeOH-pos-001.d",
"F:/Hugo/MNX-neg-001.d"]

msconvert_filepath="C:\Program Files\ProteoWizard 3.0.23286.bde6557 64-bit\msconvert.exe"

#method=' --mzML --filter "peakPicking vendor" --filter "zeroSamples removeExtra" --filter "titleMaker Run: <RunId>, Index: <Index>, Scan: <ScanNumber>"'
method=' --mzML --filter "titleMaker Run: <RunId>, Index: <Index>, Scan: <ScanNumber>"' #no peak picking


output_folder="C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml" #if "", write inside of file location


if len(output_folder):

    if not os.path.exists(output_folder): os.makedirs(output_folder)

for folder in folders:
    
    out_path=output_folder
    if not len(out_path):
        out_path=folder
    

    command="cd" +' "'+out_path+'" && '+' "'+msconvert_filepath+'" '
    command+='"'+folder+'"' 
    command+=method+' -o '+'"'+out_path+'"'

    print(command)
    stdout, stderr =subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    print(stderr)
