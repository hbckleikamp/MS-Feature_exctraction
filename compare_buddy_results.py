# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:34:20 2024

@author: hkleikamp
"""

#%% change directory to script directory (should work on windows and mac)
import os
from pathlib import Path
from inspect import getsourcefile
os.chdir(str(Path(os.path.abspath(getsourcefile(lambda:0))).parents[0]))
basedir=os.getcwd()
print(basedir)


#%% modules
import numpy as np
import pandas as pd
from collections import Counter



#%% Inputs


PSMs_file="E:/Proteomics_Jeanine/MSFragger_open_single_strain/B004241_Bp_PRC-2296_Remaut_shotgun_1_20180722184121.pin"

#utils files
unimod_file=str(Path(basedir,"utils","unimod_parsed.txt"))                       # elemental compositions of modifications
isotope_file=str(Path(basedir,"utils","natural_isotope_abundances.tsv"))         # nist natural isotope abundances
amino_acids_file=str(Path(basedir,"utils","AA_formulas.txt"))                    # elemental compositions of amino acids
calibrants_file=str(Path(basedir,"utils","calibrants.tsv")) # internal standards used for calibration (DOI: 10.1016/j.aca.2008.04.043)


#%% params

neighbouring_scans=10
min_charge=1
max_charge=6
charges=np.array([i for i in range(min_charge,max_charge+1)])
isotope_window=np.array([0,1,2,3,4,5]) 
ppm=20
group="max"
minimum_isotopes=1

#%% fixed variables

charge_mass    =1.007276466621     #proton_mass (or put electron mass here in case of negative charge)
# isotope_mass   =1.002881202641     #1.0033548350722992=mass shift of C13-C12 (this actually depends on the chemical composition!!! for example isotopes H2-1: 1.00627, N15-14: 0.99699)

#%% Load and prep auxiliary files

unimod_df,isotope_table,aa_comp,calibrants=[pd.read_csv(i,sep="\t",index_col=[0]) for i in [unimod_file,isotope_file,amino_acids_file,calibrants_file]]

#parse amino acid composition
aa_elements=aa_comp.columns[1:].tolist()
for i in ["","N-term","C-term"]:              #add compositions for missing, N-term C-term (zero cases)
    aa_comp.loc[i,:]=[0]*len(aa_comp.columns) 

#parse unimod information
unimod_elements=unimod_df.columns[3:].tolist()
unimod_df.loc["",:]=["",0,""]+[0]*len(unimod_elements) #add zero cases

#parse isotopic data
element_mass=isotope_table[isotope_table["Standard Isotope"]][["symbol","Relative Atomic Mass"]].set_index("symbol")
elements=isotope_table.symbol.drop_duplicates().tolist()
standard_isotopes=isotope_table.loc[isotope_table["Standard Isotope"],["symbol",'Isotopic  Composition']].set_index("symbol")
isotope_table["mass_number"]=isotope_table["mass_number"].astype(int)
isotope_mass=isotope_table.loc[isotope_table["symbol"]=="C","Relative Atomic Mass"].diff().values[-1]

#linearize isotopic window (for fast fourier transform)
isotopic_range=np.arange(isotope_table["delta neutrons"].min(), isotope_table["delta neutrons"].max()+1)
df=pd.DataFrame(np.vstack([np.array(list(zip([i]*len(isotopic_range),isotopic_range))) for i in isotope_table["symbol"].drop_duplicates().values]),columns=["symbol","delta neutrons"])
df["delta neutrons"]=df["delta neutrons"].astype(float).astype(int)
l_iso=isotope_table.merge(df,on=["symbol","delta neutrons"],how="outer").fillna(0)
l_iso=l_iso[['symbol','Isotopic  Composition','delta neutrons']].pivot(index="symbol",columns="delta neutrons",values="Isotopic  Composition")
l_iso[list(range(int(max(l_iso.columns)+1),int(max(l_iso.columns)+1+256-len(l_iso.columns))))]=0 #right-pad with to reach 256 sampling points (needs to be power of 2, but more padding reduces instability) 
l_iso=l_iso[l_iso.columns[l_iso.columns>=0].tolist()+l_iso.columns[l_iso.columns<0].tolist()]    #fftshift
l_iso.columns=l_iso.columns.astype(int)

#parse calibrants mass
cs=calibrants["Mz"].drop_duplicates().sort_values()


#%% Functions
def read_unknown_delim(file,Keywords=["Peptide"]):

    with open(file, "r") as f:
        lines=f.readlines()    
    header=lines[0].replace("\n","")#.split("\t")
    delims=[i[0] for i in Counter([i for i in header if not i.isalnum()]).most_common()]

    for d in delims:
        if not set(Keywords)-set(header.split("\t")):
            header=header.split(d)
            df=pd.DataFrame([i.replace("\n","").split("\t",len(header)-1) for i in lines[1:]],columns=header)
            return df[[i for i in header if i!=""]].drop_duplicates()
        

def EleCounter(psms, # dataframe, requires a column titled "Peptide"

               decimal=".",       # "." or "," (dynamically detected)
               description="mass", # "mass" : mass of modification is listed or "name" : names are exactly as unimod names (dynamically detected)
               delta="auto",      # in case of mass, is it AA+mod (True) or just mass of mod (False) (auto: dynamically detected)
                ):
    print("Calculating Elemental Composition")

    #parse modifications
    ismod=~psms["Peptide"].str.isalnum()
    mod_peps=psms.loc[ismod,"Peptide"]
    cs=Counter(psms["Peptide"].sum())

    if mod_peps.str.count(",").sum()>mod_peps.str.count(".").sum(): decimal=","     # get decimal
    delimiters=[k for k,v in cs.items() if not k.isalnum() and k!=decimal]          # get delim
    if  cs.get(delimiters[0])!=cs.get(decimal): description="name"                  # get modification type (name or mass)
    
    for d in delimiters: mod_peps=mod_peps.str.replace(d,"_").str.rstrip("_")
    mod_peps[mod_peps.str.startswith("_")]="n"+mod_peps[mod_peps.str.startswith("_")]  # add N-Term
    s=mod_peps.str.split("_")
    mod_aas= s.apply(lambda x: "".join(x[0::2]))
    mod_pairs=pd.concat([s.apply(lambda x: x[1::2]).explode(),
                         s.apply(lambda x: [i[-1]for i in x[0:-1:2]]).explode()],axis=1)
    mod_pairs.columns=["mod","site"]
    mod_pairs["site"]=mod_pairs["site"].str.replace("n","N-term")
    um=mod_pairs.drop_duplicates()
    
    #add compositions from unimod based on description
    if description=="mass": #based on mass
        mod_pairs["mod"]=mod_pairs["mod"].astype(float)
        um["mod"]=um["mod"].astype(float)
        if delta=="auto": delta=(mod_pairs["mod"]-aa_comp["Mass"].loc[mod_pairs["site"].tolist()].values).sum()>0 #detect if modification masses include the amino acid
        if delta:         um["mod"]=um["mod"]-aa_comp.loc[um["site"]].values
        um=um.merge(unimod_df,on="site",how="left")                                         #merge with unimod
        um["md"]=abs(um["mod"]-um["delta_mass"])                                            #calc mass difference
        um=um.sort_values(by=["mod","site","md"]).groupby(["mod","site"],sort=False).nth(0) #pick lowest mass difference
    else: #based on name (has to match exactly)
        um=um.merge(unimod_df,left_on="mod",right_on="full_name",how="left")
    
    mod_comps=mod_pairs.reset_index().merge(um[["mod","site"]+unimod_elements],on=["mod","site"],how="left")[["index"]+unimod_elements]
    unmod=pd.concat([psms.Peptide[~ismod],mod_aas]).reset_index()
    unmod["Peptide"]=unmod["Peptide"].apply(list)
    unmod_comps=unmod.explode("Peptide").merge(aa_comp,left_on="Peptide",right_index=True,how="left")[["index"]+aa_elements]
    comps=pd.concat([unmod_comps,mod_comps]).fillna(0).groupby("index").sum()
    comps["H"]=comps["H"]+2 #add back water loss
    comps["O"]=comps["O"]+1
    s=comps.sum() #remove 0 columns
    comps=comps[s[s>0].index.tolist()] 
    comps["mass"]=(comps*element_mass.loc[comps.columns].values.T).sum(axis=1) # add_mass
    
    return pd.concat([psms,comps],axis=1)
 

def IsotopePredictor(psms,                      # dataframe, requires element columns
                     minimum_value=10**-5,      #minimum relative abundance of isotopic peak to be reported (otherwise output would have 256 columns). 
                     normalize=True,
                     add_psms=False,
                     ):
    
    
    
    #compute fft baseline
    elements=psms.columns[psms.columns.isin(l_iso.index)].tolist()
    one=np.ones([len(psms),len(l_iso.columns)])*complex(1, 0)
    for e in elements:
        one*=np.fft.fft(l_iso.loc[e,:])**psms[e].values.reshape(-1,1)
    baseline=pd.DataFrame(np.fft.ifft(one).real,columns=[i for i in l_iso.columns])
    baseline=baseline[((baseline>minimum_value).any()[(baseline>minimum_value).any()]).index]
    baseline=baseline[baseline.columns.sort_values()]
    
    if len(isotope_window): baseline=baseline[isotope_window]
    if normalize:           baseline=baseline.divide(baseline.sum(axis=1),axis=0)
    
    baseline.columns=["theoretical_isotope_"+str(i) for i in baseline.columns]
    if add_psms: baseline=pd.concat([psms,baseline],axis=1)    

    return baseline
#%% 1. Read PSMs file

#1a read PSMs file (pin file)

FDR_cutoff=0.05
score_column="hyperscore"
decoy_delimiter="rev_"



psms=read_unknown_delim(PSMs_file)  #tsv file with required columns: scan, Peptide 
if PSMs_file.endswith(".pin"):      #pin specfic prep
    psms["scan"]=psms["ScanNr"]
    psms["Peptide"]=psms["Peptide"].str.split(".").apply(lambda x: ".".join(x[1:-1]))

# 1b calculate chemical composition
psms=EleCounter(psms)
p_elements=[i for i in psms.columns if i in elements]

# # 1c calculate theoretical isotope envelope
# print("Predicting Natural Isotopic Abundance")
# baseline=IsotopePredictor(psms)
# psms=pd.concat(psms,baseline)

# #%% Load protein isotopologue

# ei=pd.read_csv("E:/Proteomics_Jeanine/mzML/B004241_Bp_PRC-2296_Remaut_shotgun_1_20180722184121_calibrated_protein_isotoplogue.tsv",sep="\t")
# ei["index"]=ei.id.str.split("_").apply(lambda x: x[0]).astype(int)
# #need it for scan information

#%% Load buddy annotated proteins

bp=pd.read_csv("C:/Metabolomics/Buddy/Buddy_python_proteins_charge_corrected.tsv",sep="\t")
#bp=bp.merge(ei["index"],left_on="id",right_index=True,how="left")
bp["index"]=bp["id"].str.split("_").apply(lambda x:x[0]).astype(int)
bp=bp.sort_values(by=["index","estimated_fdr"])
bp=bp.rename(columns={"mass":"buddy_mass"})
#gbp=bp.groupby("index",sort=False).nth(0).reset_index(names="idx")
gbp=bp.copy()

#%%


#shared elements
#ce=set([i for i in bp.columns.tolist()+psms.columns.tolist() if i in isotope_table.symbol.drop_duplicates().tolist()])

ce=["C","H","N","O","P","S"]

for i in ce:
    if i not in psms.columns:
        psms[i]=0
    if i not in gbp.columns:
        gbp[i]=0


ss=[]
for i in ce:
    s=psms[i].astype(int).astype(str)
    e=pd.Series([i]*len(psms))
    e.loc[s=="0"]=""
    ss.append(e+s.replace("1","").replace("0",""))
psms["theoretical_composition"]=pd.concat(ss,axis=1).sum(axis=1)

ss=[]
for i in ce:
    s=gbp[i].astype(int).astype(str)
    e=pd.Series([i]*len(gbp))
    e.loc[s=="0"]=""
    ss.append(e+s.replace("1","").replace("0",""))
gbp["buddy_composition"]=pd.concat(ss,axis=1).sum(axis=1)


psms['ExpMass']=psms['ExpMass'].astype(float)
psms=psms[(abs(psms['ExpMass']-psms["mass"])/psms['ExpMass']*1000000)<20]

m=psms[["theoretical_composition",'Label','ExpMass','hyperscore']].merge(gbp,left_index=True,right_on="index",how="inner")
m["mass_shift"]=abs(m['ExpMass']-m["buddy_mass"])/m['ExpMass']*1000000

eq=m[m["buddy_composition"]==m["theoretical_composition"]]
neq=m[m["buddy_composition"]!=m["theoretical_composition"]]
neq=neq[neq.id.isin(eq.id)]

#%%
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
eq["mass_shift"].plot.hist(bins=60,color=(0, 0.5, 0, 0.3))
plt.xlim(0,10)
fig,ax=plt.subplots()
neq["mass_shift"].plot.hist(bins=60,color=(0.5, 0, 0, 0.3))
plt.xlim(0,10)
#%%
fig,ax=plt.subplots()
eq['ms1_isotope_similarity'].plot.hist(bins=100,color=(0, 0.5, 0, 0.3))
plt.xlim(0.5,1)

fig,ax=plt.subplots()
neq['ms1_isotope_similarity'].plot.hist(bins=100,color=(0.5, 0, 0, 0.3))
plt.xlim(0.5,1)



#%% Random forest classification

sb=pd.read_csv("C:/Metabolomics/Buddy/Buddy_python_OpenMS_charge_corrected.tsv",sep="\t")
sb["measured_mass"]=sb["measured_mass"]-charge_mass
sb["mass_shift"]=abs(sb["measured_mass"]-sb["buddy_mass"])/sb["measured_mass"]*1000000




#%%

t=sb[(sb["mass_shift"]<2) & (sb['ms1_isotope_similarity']>0.9) ]
te=t[(t["N"]==4) & (t["O"]==4)]

fig,ax=plt.subplots()
g=t.groupby("S").size().plot.bar(stacked=True,rot=0)
# plt.xticks( rotation='vertical')
# plt.show()

#%%

fig,ax=plt.subplots()
plt.stem(res.iloc[11][mcols],res.iloc[11][icols])
plt.xlim(596.630962-0.1,596.630962+3)

fig,ax=plt.subplots()
plt.stem(res.iloc[21][mcols],res.iloc[21][icols])
plt.xlim(792.399878-0.1,792.399878+3)

#%%
fig,ax=plt.subplots()
plt.stem(res.iloc[74][mcols],res.iloc[74][icols])
plt.xlim(1206.520798-0.1,1206.520798+4)

fig,ax=plt.subplots()
plt.stem(res.iloc[76][mcols],res.iloc[76][icols])
plt.xlim(603.764037-0.1,603.764037+4)