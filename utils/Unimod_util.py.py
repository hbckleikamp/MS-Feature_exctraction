# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:15:26 2023

@author: ZR48SA
"""

#%% change directory to script directory (should work on windows and mac)
import os
from pathlib import Path
from inspect import getsourcefile
os.chdir(str(Path(os.path.abspath(getsourcefile(lambda:0))).parents[0]))
print(os.getcwd())

#%%
import urllib


import pandas as pd




#%% download
print("Downloading unimod file")
KEGGurl="https://www.unimod.org/xml/unimod.xml"
urllib.request.urlretrieve(KEGGurl, "unimod.xml")

#%%
print("Parsing unimod file")


with open("unimod.xml", "r") as f:
    xmlstring=f.read()
    

elements=[i.split('"')[0] for i in xmlstring.split('<umod:elem title="')[1:]]

mods=[i.split("</umod:mod>")[0] for i in xmlstring.split("<umod:mod ")[1:]]

parsed=[]
for ix,mod in enumerate(mods):
    print(ix)
  

    title=mod.split('title="')[1].split('"')[0]
    full_name=mod.split('full_name="')[1].split('"')[0]
    
    d=mod.split('umod:delta mono_mass="')[1]
    delta_mass=d.split('"')[0]
    composition=[d.split('element symbol="'+e+'" number="')[1].split('"')[0] if 'element symbol="'+e+'" number="' in d else 0 for e in elements]
    site=[i.split('"')[0] for i in mod.split('site="')[1:]]
    parsed.append([title,full_name,delta_mass,site]+composition)

parsed_df=pd.DataFrame(parsed,columns=["title","full_name","delta_mass","site"]+elements).set_index("title")
parsed_df.index=parsed_df.index.str.replace('&gt;','>') #html parsing
parsed_df.loc["",:]=["0"]*len(parsed_df.columns) #add zero row
parsed_df[elements]=parsed_df[elements].replace("title=","0").astype(int)
parsed_df["delta_mass"]=parsed_df["delta_mass"].astype(float)

parsed_df=parsed_df.explode("site").drop_duplicates()

parsed_df.to_csv("unimod_parsed.txt",sep="\t")