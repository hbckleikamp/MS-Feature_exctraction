# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:18:00 2024

@author: hkleikamp
"""


#%% change directory to script directory (should work on windows and mac)
import os
from pathlib import Path
from inspect import getsourcefile
os.chdir(str(Path(os.path.abspath(getsourcefile(lambda:0))).parents[0]))
basedir=os.getcwd()
print(basedir)

#%% Modules

import pandas as pd
import numpy as np

#%% Function

def parse_form(form): #chemical formular parser
    e,c="",""
    
    es=[]
    cs=[]
    for i in form:
        
        if i.isupper(): #new entry   
            
            if e: 
                if not c: c="1"
                es.append(e)
                cs.append(c)
            e=i         
            c=""
            
        elif i.islower(): e+=i
        elif i.isdigit(): c+=i
            
    
    if e: 
        if not c: c="1"
        es.append(e)
        cs.append(c)
    
    return pd.DataFrame(list(zip(es,cs)),columns=["elements","counts"]).set_index("elements").T

#%% PEG/PPG 


adducts="H+","Na+","K+"
#neutral_losses

#%% polycyclosiloxane


#NH4 adduct, #-CH4 loss

#%% Phthalates

names=[]
forms=[]
f="C:/Wout_features/utils/phthalates.txt"
with open(f,"r") as o:
    lines=o.readlines()

for line in lines:    
    names.append(" ".join(line.split(" ")[0:-5]))
    form=parse_form(line.split(" ")[-5])
    

    
    
    forms.append(form)
    
df=pd.concat(forms).reset_index(drop=True)
df["name"]=names
df=df[['name','C', 'H', 'O']]
df=df.set_index("name")
df.to_csv("phthalates.tsv",sep="\t")

#%% 


    