# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:46:47 2023

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

import matplotlib.pyplot as plt

import pyopenms


#%% Inputs


PSMs_file="E:/Proteomics_Jeanine/MSFragger_open_single_strain/B004241_Bp_PRC-2296_Remaut_shotgun_1_20180722184121.pin"
mzML_file='E:/Proteomics_Jeanine/mzML/B004241_Bp_PRC-2296_Remaut_shotgun_1_20180722184121.mzML'
mzML_file="E:/Proteomics_Jeanine/mzML/B004241_Bp_PRC-2296_Remaut_shotgun_1_20180722184121_calibrated.mzML"
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




#%% 2. Read mzML

print("reading mzML")
exp = pyopenms.MSExperiment()
pyopenms.MzMLFile().load(mzML_file, exp)
ms1=[s for s in exp.getSpectra() if s.getMSLevel()==1]
ms1_scans=np.array([spec.getNativeID().split("scan=")[1] for spec in ms1]).astype(int)


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

# 1c calculate theoretical isotope envelope
print("Predicting Natural Isotopic Abundance")
baseline=IsotopePredictor(psms)



# 1d calculate exact isotopic mass
esis=[]
qs=[] #quotients
for e in p_elements:
    nes=[i for i in p_elements if i!=e]
    esi=IsotopePredictor(psms[nes])
    qs.append(esi["theoretical_isotope_0"]-baseline["theoretical_isotope_0"])
    esis.append(esi)

qs=pd.concat(qs,axis=1)
qs.columns=p_elements
nqs=qs.divide(qs.sum(axis=1),axis=0)

s=np.zeros(baseline.shape)
m=np.zeros(baseline.shape)
fs=[]
for ix,esi in enumerate(esis):
    e=p_elements[ix]
    val=standard_isotopes.loc[e][0]**psms[e]
    f=baseline-esi.multiply(val,axis=0)#.reshape(-1,1)
    s=s+f.values
    fs.append(f)

for ix,f in enumerate(fs):
    e=p_elements[ix]
    nf=f.divide(s,axis=0)

    i=isotope_table.loc[isotope_table["symbol"]==e]
    d=i[["Relative Atomic Mass","mass_number"]].diff().values[-1]
    im=d[0]/d[1]
    m+=im*nf

isotope_masses=m*isotope_window


#%% 3. Find neighbouring scans

ms2_scans=psms["scan"].astype(int) #.sort_values()
inc,c,x=0,0,0
neighbours,ds,m1s,uscans=[],[],[],[]

for m2 in ms2_scans:
    for m1 in ms1_scans[x:]:
        d=m1-m2
        ds.append(abs(d))
        m1s.append(m1)
        if c>=neighbouring_scans:
            if d>ds[c-neighbouring_scans]:
                nscans=m1s[-neighbouring_scans:]
                neighbours.append(nscans)
                uscans.extend(nscans)
                x+=len(m1s)-neighbouring_scans-1
                ds,m1s,c=[],[],0
                break
        c+=1    
uscans=list(set(uscans))  
uscans.sort()

s=pd.Series(ms1_scans)
ms1_peaks=[np.vstack(ms1[i].get_peaks())  for i in s[s.isin(uscans)].index]
masses     =[i[0] for i in ms1_peaks]
intensities=[i[1] for i in ms1_peaks]
#separate masses and intensities outside or in loop?


#%% 4. Feature extraction

## Extract PSMs
#this loop assumes ms1scans are sorted
icols,mcols,lcols,ucols,ppm_cols=[[str(i)+x for i in isotope_window] for x in ["_isotope","_mass","_lower","_upper","ppm_shift"]] #column names
charge_window=np.arange(min_charge,max_charge+1)



# groupby mass, neighbours
ndf=psms[["mass"]]
ndf[mcols]=ndf[["mass"]].values+isotope_masses  #isotope_window*isotope_mass
ndf["neighbours"]=neighbours
ndf=ndf.explode("neighbours")  #.groupby("neighbours") #but we need original peptides?


gs=pd.concat([ndf for i in charge_window])
gs["Charge"]=np.hstack([np.ones(len(ndf))* i for i in charge_window])
#gs[mcols]=gs[["mass"]].values+isotope_window*isotope_mass
gs[mcols]=(gs[mcols]+charge_mass*gs[["Charge"]].values)/gs[["Charge"]].values
gs[lcols]=gs[mcols]*(1-ppm/1000000)
gs[ucols]=gs[mcols]*(1+ppm/1000000)
gs[icols]=0

ndf=gs.groupby("neighbours")
#outside loop, rename variables

counter=0
res=[]
for n,gs in ndf:
    
    Mass,Intensity=masses[counter],intensities[counter]
    counter+=1
    
    if counter%1000==0:
        
        print("fraction completed: "+str(round(counter/len(uscans),3)))
    
    
    l=gs[lcols].values.flatten()          #lower limits 
    f=gs[ucols].values.flatten()          #upper limits  
    u=np.sort(f)                         #sorted upper limits
    l=np.sort(l)                         #sorted upper limits
    ua=np.argsort(f)                     #index of sorted upper limits
    
    
    ims,inc=[],0 
    for im,m in enumerate(Mass):
        
        #if mass is higher than current upper limit, more to next upper limit
        while m>u[inc]: 
            inc+=1
            if inc==len(u):
                break

        if inc==len(u):
            break
        
        #temporary increment
        c=0
        while (m>l[inc+c]) & (m<u[inc+c]): 
          
            ims.append([inc+c,im])
            c+=1
            if inc+c==len(u):
                break
       
    ims=np.array(ims)
    
    
    if len(ims): 
        s=gs[icols].shape
        z0=np.zeros(s)
        incs=ims[:,0]
        
        
        #map back ints
        ints=Intensity[ims[:,1]]
        
        #incase of multiple masses withtin the ppm window pick either maximum or sum of intensities
        u_incs,inv=np.unique(incs,return_inverse=True) 
        if len(incs)-len(u_incs):
            if group=="sum":
                ints=np.bincount(inv, weights=ints) 
            if group=="max":
                out=np.zeros(u_incs.shape)
                #with np.errstate(invalid='ignore'):
               
                np.maximum.at(out, inv, ints)
                ints=out
            incs=u_incs
   
        z0[np.unravel_index(ua[incs],s)]=ints
        gs[icols]=z0
        
        #map back masses
        m_mass=np.bincount(inv, weights=Mass[ims[:,1]])/np.array([i for i in Counter(inv).values()]) 
        zm=np.zeros(s)
        zm[np.unravel_index(ua[incs],s)]=m_mass
        zm[zm==0]=np.nan
        gs[ppm_cols]=(gs[mcols]-zm)/zm*1000000 
        
        gs=gs[(gs[icols]>0).sum(axis=1)>=minimum_isotopes] #minimum isotope filtering
        res.append(gs.drop(lcols+ucols,axis=1))
   

res=pd.concat(res).fillna(0)
res["mz"]=(res["mass"]+charge_mass*res["Charge"])/res["Charge"]

#res=res.reset_index(names="Scan").sort_values(by=["Scan","Charge"])
#%% Isotope filtering

from numpy.linalg import norm
def IsotopeFilter(its,
                       minimum_cosine_similarity=0.0, # filter on minimum cosine similarity of isotope to monoisotopic peak (between 0-1)
                       default="monoisotopic"       , # which peak to default to for cosine similarity filtering (monoisotopic, most_intense, first_most_nonzero)
                       minimum_isotopes=3           , # filter on minimum isotopes that should be left after cosine similarity filtering
                       minimum_intensity=0          , # filter on minimum total intensity of isotopic peaks
                       top_intensity_fraction=0.7   , # filter on minimum intensity fraction of most intense neighbouring scan
                       remove_gapped=False          , # remove gapped isotope patterns that miss one or more isotopes in the middle of the detected envelope 
                       combine=False                , # sum intensities of neighbours                 
                       normalize=True):               # normalize intensities



    #cosine similarity filtering
    if minimum_cosine_similarity>0:
        print("cosine similarity filtering")
        ux=its["index"].nunique()
        groups=its.groupby("index")
        gs,ixs=[],[]
        c=0
        for _,g in groups:
            c+=1
            if c%10000==0:
    
                print("fraction completed: "+str(round(c/ux,3)))

            #which peak to default to for cosine similarity filtering?
            if default=="biggest_drop":                      #biggest drop biggest drop in intensity, should work better with co-eluting peptides
                A=g[icols[g[icols].sum().diff().argmin()-1]]
            elif default=="first_most_nonzero":              #if you pick first most nonzero you might select more background noise?
                A=g[(g[icols]>0).sum().idxmax()] 
            elif default=="monoisotopic":                    #if you pick monoisotopic, you will detect less at higher labelling and longer peptides= 
                A=g[icols[0]] 
            elif default=="most_intense":
                A=g[g[icols].sum().idxmax()]                 #if you pick most intense you might select more contamination from co-eluting peptides?
                
                
            nA=norm(A)
            if nA:
                cosines=[]
                for i in icols:
                    B=g[i]
                    nB=norm(B)
                    if nB:
                        cosines.append(np.dot(A,B)/(nA*nB))
                    else:
                        cosines.append(0)
                        
                g[[icols[ix] for ix,i in enumerate(cosines) if i <minimum_cosine_similarity]]=0
                gs.append(g.values)
                ixs.extend(g.index)

        gs=np.vstack(gs)
        its=pd.DataFrame(gs,columns=its.columns,index=ixs)
        # its=pd.concat(gs)
        print("done")

    its=its[~((its[icols]>0).sum(axis=1)<minimum_isotopes)] #post filter                                                                   # filter on minimum number of isotopic peaks
    its=its[~(its[icols].sum(axis=1)<minimum_intensity)]                                                                                   # filter on minimum total intensity
    if remove_gapped: its=its[its[icols].apply(np.flatnonzero,axis=1).apply(np.diff).apply(max)==1]                                        # remove gapped isotopic envelopes
    if combine:       its=its.groupby("index").sum()
    if normalize:     its[icols]=its[icols].divide(its[icols].sum(axis=1),axis=0)

    return its

fil_res=res.copy().reset_index()
fil_res["index"]=fil_res["index"].astype(str)+"_"+fil_res.Charge.astype(str)
fil_res=fil_res[["index"]+icols]
fil_res=IsotopeFilter(fil_res,remove_gapped=True,combine=False,minimum_cosine_similarity=0.8)
#%%
ri=res.reset_index(names="scan")
[ri.pop(i) for i in icols]
ri=ri.merge(fil_res,left_index=True,right_index=True,how="right").set_index("scan")
res=ri.copy()
#%%

### homogenize columns
n=res[icols].divide(res[icols].sum(axis=1),axis=0) #normalized
n.columns=[int(i.split("_")[0]) for i in n.columns]

theor=baseline.copy()
theor.columns=[int(i.replace("theoretical_isotope_","")) for i in theor.columns ]
theor[list(set(n.columns)-set(theor.columns))]=0 #add 0 to theorerical isotopes for each missing column
theor=theor.loc[n.index,n.columns]        #make sure that theor has the same columns as measured

ztheor=np.where(n==0,0,theor)                       
ztheor=ztheor/ztheor.sum(axis=1).reshape(-1,1)          #renormalize theoretical envelope
dtheor=n-ztheor  #you subtract to find the 

dtheor.columns=["isotope_diff_"+str(i) for i in dtheor.columns]
res["mean_ed"]=abs(dtheor).mean(axis=1).values


sres=res.groupby("index").mean()
sres=sres.reset_index(names="id")
sres=sres.rename(columns={"mz":'monoisoMz',"Charge":"charge"})

sres.to_csv(mzML_file.replace(Path(mzML_file).suffix,'_protein_isotoplogue.tsv'),sep="\t")

#%%



#

# #%%
# fres=res[res["mean_ed"]>0]


# target_scans,decoy_scans=psms.loc[psms.Label=="1","scan"].astype(int),psms.loc[psms.Label=="-1","scan"].astype(int)

# us=np.unique(fres.index)
# ftarget_scans=target_scans[target_scans.isin(us)]
# fdecoy_scans=decoy_scans[decoy_scans.isin(us)]


# #%% euclidian distance to isotopic envelope

# fig,ax=plt.subplots()
# fres.loc[ftarget_scans,"mean_ed"].plot.hist(bins=100)
# fig,ax=plt.subplots()
# fres.loc[fdecoy_scans,"mean_ed"].plot.hist(bins=100)

# #%%

# #number of detected features
# t=fres.loc[ftarget_scans]
# d=fres.loc[fdecoy_scans]

# fig,ax=plt.subplots()
# t.groupby(t.index).size().plot.hist(bins=30)
# plt.xlim(0,30)
# fig,ax=plt.subplots()
# d.groupby(d.index).size().plot.hist(bins=30)
# plt.xlim(0,30)
# #theres a bunch of questions
# #1. does this weird isotope mass improve?
# #2. Does calibration improve number of hits (slight decrease in len res, so worse??)
# #2.  What differentiates 
# #%%