# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:26:52 2023

@author: ZR48SA
"""

#%% change directory to script directory (should work on windows and mac)
import os
from pathlib import Path
from inspect import getsourcefile
os.chdir(str(Path(os.path.abspath(getsourcefile(lambda:0))).parents[0]))
print(os.getcwd())


import requests
import pandas as pd
import numpy as np


url="https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl"
tables = pd.read_html(url)[0]


#remove fully nan rows and columns
tables=tables[~tables.isnull().all(axis=1)]
tables=tables[tables.columns[~tables.isnull().all(axis=0)]]


tables.columns=["atomic_number","symbol","mass_number",'Relative Atomic Mass',
       'Isotopic  Composition', 'Standard Atomic Weight', 'Notes']
tables=tables[["atomic_number","symbol","mass_number",'Relative Atomic Mass',
       'Isotopic  Composition', 'Standard Atomic Weight']]

tables.loc[tables["atomic_number"]==1,"symbol"]="H" #remove deuterium and tritium trivial names
tables=tables[tables['Isotopic  Composition'].notnull()].reset_index(drop=True)


#floatify mass and composition
for i in ['Relative Atomic Mass',
       'Isotopic  Composition', 'Standard Atomic Weight']:
    tables[i]=tables[i].str.replace(" ","").str.replace("(","").str.replace(")","").str.replace(u"\xa0",u"")
tables[['Relative Atomic Mass', 'Isotopic  Composition']]=tables[['Relative Atomic Mass', 'Isotopic  Composition']].astype(float) 
tables['Standard Atomic Weight']=tables['Standard Atomic Weight'].str.strip("[]").str.split(",")

#add boolean flag for baseline isotope
etables=tables.explode('Standard Atomic Weight').fillna(1000)
etables['Standard Atomic Weight']=etables['Standard Atomic Weight'].astype(float)
groups=etables.groupby("symbol",sort=False)
bs=[]
for n,g in groups:
    b=np.array([False]*len(g))
    d=abs(g[["Relative Atomic Mass"]].values-   g[['Standard Atomic Weight']].values.reshape(1,-1))
    b[np.unravel_index(d.argmin(),d.shape)[0]]=True
    bs.extend(list(b))
etables["Standard Isotope"]=bs



tables["Standard Isotope"]=etables.groupby(["symbol",'mass_number'],sort=False)["Standard Isotope"].sum().astype(bool).values
tables["delta neutrons"]=pd.concat([x["mass_number"]-x[x["Standard Isotope"]]["mass_number"].values for n,x in tables.groupby("symbol")])
tables.to_csv("natural_isotope_abundances.tsv",sep="\t")