# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:08:32 2024

@author: e_kle
"""


#%% Modules


from collections import Counter
import pyopenms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #testing
from sklearn.cluster import DBSCAN
from pathlib import Path
#%%


#Run dinosaur

#%%

mzML_file="C:/Proteomics_Jeanine/mzML/B004241_Bp_PRC-2296_Remaut_shotgun_1_20180722184121.mzML"
featurefile="C:/Proteomics_Jeanine/mzML/B004241_Bp_PRC-2296_Remaut_shotgun_1_20180722184121_openms_features.tsv"

neighbouring_scans=10
ppm=20
group="max"
minimum_isotopes=1
isotope_window=np.array([0,1,2,3,4,5]) 

min_charge,max_charge=1,6 #need charge expansion for unknown charge traces in openms

charge_mass    =1.007276466621     #proton_mass (or put electron mass here in case of negative charge)
isotope_mass   =1.002881202641     #1.0033548350722992=mass shift of C13-C12 (this actually depends on the chemical composition!!! for example isotopes H2-1: 1.00627, N15-14: 0.99699)


#%%
#output type: buddy or sirius? 
#max mass limit

#%% Read Dinosaur file

dino=pd.read_csv(featurefile,sep="\t")

#expand unknown charges from openMS
q=dino["charge"]==0
z,nz=dino[q],dino[~q]
zs=[]
for c in np.arange(min_charge,max_charge+1):
    zc=z.copy()
    zc["charge"]=c
    zs.append(zc)
dino=pd.concat([nz]+zs)


dino=dino.sort_values(by="rtApex")
dino_RTs=dino.rtApex


#%% 2. Read mzML

print("reading mzML")
exp = pyopenms.MSExperiment()
pyopenms.MzMLFile().load(mzML_file, exp)
ms1=[s for s in exp.getSpectra() if s.getMSLevel()==1]
ms1_scans=np.array([spec.getNativeID().split("scan=")[1] for spec in ms1]).astype(int)
ms1_RTs=np.array([spec.getRT() for spec in ms1]) #important: dinosaur rt is in minutes not in seconds, while openMS is in seconds

conv=pd.Series(ms1_scans)
conv.index=ms1_RTs


#%% find nearest scans based on retention time




inc,c,x=0,0,0
neighbours,ds,m1s,uscans=[],[],[],[]

for m2 in dino_RTs:
    nscans=[]
    for m1 in ms1_RTs[x:]:
        d=m1-m2
        ds.append(abs(d))
        m1s.append(m1)
        if c>=neighbouring_scans:
            if d>ds[c-neighbouring_scans]:
                break
        c+=1


    nscans=m1s[-neighbouring_scans:]
    neighbours.append(nscans)
    uscans.extend(nscans)
    x+=len(m1s)-neighbouring_scans-1 #WTF is dit?
    ds,m1s,c=[],[],0    


    
uscans=list(set(uscans))  
uscans=conv.loc[uscans].tolist()
uscans.sort()

s=pd.Series(ms1_scans)
ms1_peaks=[np.vstack(ms1[i].get_peaks())  for i in s[s.isin(uscans)].index]
masses     =[i[0] for i in ms1_peaks]
intensities=[i[1] for i in ms1_peaks]

#%% Feature extraction


## Extract PSMs
#this loop assumes ms1scans are sorted
icols,mcols,lcols,ucols,ppm_cols=[[str(i)+x for i in isotope_window] for x in ["_isotope","_mass","_lower","_upper","ppm_shift"]] #column names


# groupby mass, neighbours
ndf=dino[["monoisoMz","charge"]]
ndf["neighbours"]=neighbours

gs=ndf.explode("neighbours")  #.groupby("neighbours") #but we need original peptides?
gs[mcols]=gs[["monoisoMz"]].values+isotope_mass/gs[["charge"]].values*isotope_window 
gs[lcols]=gs[mcols]*(1-ppm/1000000)
gs[ucols]=gs[mcols]*(1+ppm/1000000)
gs[icols]=0
gs["neighbours"]=conv.loc[gs.neighbours].tolist()
ndf=gs.groupby("neighbours")


#%%
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
        res.append(gs.drop(mcols+lcols+ucols,axis=1))
   

res=pd.concat(res).fillna(0)


#%%%

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
        print("done")

    its=its[~((its[icols]>0).sum(axis=1)<minimum_isotopes)] #post filter                                                                   # filter on minimum number of isotopic peaks
    its=its[~(its[icols].sum(axis=1)<minimum_intensity)]                                                                                   # filter on minimum total intensity
    if remove_gapped: its=its[its[icols].apply(np.flatnonzero,axis=1).apply(np.diff).apply(max)==1]                                        # remove gapped isotopic envelopes
    if combine:       its=its.fillna(0).groupby("index")[icols].sum()
    if normalize:     its[icols]=its[icols].divide(its[icols].sum(axis=1),axis=0)

    return its


fres=res.copy().reset_index()
fres=fres[fres['0_isotope']>0] #must have monoisotopic isotope
fres=fres[fres['2_isotope']>0] #must have 2nd isotope
fres=IsotopeFilter(fres,minimum_cosine_similarity=0.7,combine=True,remove_gapped=True)

#%% Merge and cluster

mdino=dino[["monoisoMz","charge"]].merge(fres,left_index=True,right_index=True,how="right") #cant merge on charge now

mdino[mcols]=mdino[["monoisoMz"]].values+isotope_window*isotope_mass
mc=mdino[mcols].values
ic=mdino[icols].values
mdino[mcols]=np.where(ic==0,0,mc)         
mdino=mdino.replace(0,np.nan)

# clustering


#test
# test=mdino[abs((mdino["monoisoMz"]-375.077)/375.077*1000000)<20].fillna(0).loc[:,["monoisoMz"]+icols]
# clustering = DBSCAN(eps=0.03, min_samples=2).fit(test)
# test["c"]=clustering.labels_
# test=test.sort_values(by=["c"]+icols)

clustering = DBSCAN(eps=0.03, min_samples=2).fit(mdino.loc[:,["monoisoMz","charge"]+icols].fillna(0))
mdino["c"]=clustering.labels_

cdino=mdino[mdino["c"]!=-1].groupby("c").mean().reset_index()
cdino=pd.concat([mdino[mdino["c"]==-1],cdino])


#clustered dinosaur features

cdino.to_csv(mzML_file.replace(".mzML",'OpenMS_isotopologue.tsv'),sep="\t")

