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

def parse_formula(form): #chemical formular parser
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
    
    return pd.DataFrame(list(zip(es,cs)),columns=["elements","counts"]).set_index("elements").T.astype(int)

def generate_polymers(name,    #name of polymer
                      base,    #base formular of polymer (1-mer)
                      mer,     #added formula
                      start=1, #starting mers
                      min_count=0, 
                      max_count=20):
    
    base_form=parse_formula(base)
    mer_form=parse_formula(mer)
    names,forms=[],[]
    for i in range(min_count,max_count):
        forms.append(base_form+mer_form*i)
        names.append(name+"_"+str(start+i))
    forms=pd.concat(forms)
    forms.index=names
    
    return forms

#%% Get molecular formulas

#Phthalates
PTH=pd.read_csv("phthalates.tsv",sep="\t",index_col=[0])

# Build polymers

# #PEG
# PEG=generate_polymers(name="PEG",
#                       base="C2H6O2",
#                       mer="C2H4O")

# #PPG
# PPG=generate_polymers(name="PPG",
#                       base="C3H8O2",
#                       mer="C3H6O")


#poly cyclo siloxanes
PCS=generate_polymers(name="cyclosiloxane",
                      base="C8H24Si4O4", #base for 4-mer cylosiloxane
                      mer="C2H6SiO",
                      start=4,
                      min_count=1,#0,       #4-20 -> 0-17
                      max_count=9 #17  reduce from 4-20 to 5-12
                      )



# poly cyclo siloxanes CH4 neutral loss
PCS_CH4_loss=PCS.copy()
PCS_CH4_loss["C"]=PCS_CH4_loss["C"]-1
PCS_CH4_loss["H"]=PCS_CH4_loss["H"]-4
PCS_CH4_loss.index=[i+"-CH4" for i in PCS.index]

#forms=pd.concat([PTH,PEG,PPG,PCS,PCS_CH4_loss]).fillna(0)

forms=pd.concat([PTH,PCS,PCS_CH4_loss]).fillna(0)

#%% Add adducts

#common H+ adduct
forms["H"]+=1

#polysiloxane NH4+ adduct
PCS_NH4_add=PCS.copy()
PCS_NH4_add["N"]=1
PCS_NH4_add["H"]=PCS_NH4_add["H"]+4
PCS_NH4_add.index=[i+"+NH4" for i in PCS.index]

forms=pd.concat([forms,PCS_NH4_add]).fillna(0).reset_index()


#%% Merge identical compositions

isotope_table=pd.read_csv("natural_isotope_abundances.tsv",sep="\t" )
element_mass=isotope_table[isotope_table["Standard Isotope"]][["symbol","Relative Atomic Mass"]].set_index("symbol")
elements=forms.columns[forms.columns.isin(element_mass.index)].tolist()
forms=forms.groupby(elements)['index'].apply(lambda x: "; ".join(x)).reset_index()

#%% Compute exact mass

mass_H         =1.00782503223
mass_proton    =1.007276           #
isotope_mass   =1.0033548350722992 #mass shift of C13-C12

#parse isotopic data

forms["Mz"]=(forms[elements]*element_mass.loc[elements].values.T).sum(axis=1)

#add charge
forms["Charge"]=1 #all adducts are charge 1+ 
mass_pos=mass_proton-mass_H
forms["Mz"]=forms["Mz"]+mass_pos*forms["Charge"]

#%% Compute isotopes

#linearize isotopic window (for fast fourier transform)
isotopic_range=np.arange(isotope_table["delta neutrons"].min(), isotope_table["delta neutrons"].max()+1)
df=pd.DataFrame(np.vstack([np.array(list(zip([i]*len(isotopic_range),isotopic_range))) for i in isotope_table["symbol"].drop_duplicates().values]),columns=["symbol","delta neutrons"])
df["delta neutrons"]=df["delta neutrons"].astype(float).astype(int)
l_iso=isotope_table.merge(df,on=["symbol","delta neutrons"],how="outer").fillna(0)
l_iso=l_iso[['symbol','Isotopic  Composition','delta neutrons']].pivot(index="symbol",columns="delta neutrons",values="Isotopic  Composition")
l_iso[list(range(int(max(l_iso.columns)+1),int(max(l_iso.columns)+1+256-len(l_iso.columns))))]=0 #right-pad with to reach 256 sampling points (needs to be power of 2, but more padding reduces instability) 
l_iso=l_iso[l_iso.columns[l_iso.columns>=0].tolist()+l_iso.columns[l_iso.columns<0].tolist()]    #fftshift
l_iso.columns=l_iso.columns.astype(int)



def IsotopePredictor(psms,                      # dataframe, requires element columns
                     minimum_value=10**-5 #minimum relative abundance of isotopic peak to be reported (otherwise output would have 256 columns). 
                     ):
    
    print("Predicting Natural Isotopic Abundance")
    
    #compute fft baseline
    elements=psms.columns[psms.columns.isin(l_iso.index)].tolist()
    one=np.ones([len(psms),len(l_iso.columns)])*complex(1, 0)
    for e in elements:
        one*=np.fft.fft(l_iso.loc[e,:])**psms[e].values.reshape(-1,1)
    baseline=pd.DataFrame(np.fft.ifft(one).real,columns=[i for i in l_iso.columns])
    baseline=baseline[((baseline>minimum_value).any()[(baseline>minimum_value).any()]).index]
    baseline=baseline[baseline.columns.sort_values()]
    baseline.columns=["theoretical_isotope_"+str(i) for i in baseline.columns]
    psms=pd.concat([psms,baseline],axis=1)

    return psms

forms=IsotopePredictor(forms)
#%%

#cyclosiloxane_6	445.12112215042964
#but should be:     445.120025


#%%

forms.to_csv("calibrants.tsv",sep="\t")