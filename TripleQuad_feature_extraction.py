# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:48:33 2024

@author: hkleikamp
"""

import pyopenms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #testing
import os
import seaborn as sns

from scipy.signal import find_peaks
from numpy.linalg import norm
from scipy.signal import savgol_filter
from pathlib import Path

#%% Steps
# 1. load mzml
# 2. extract mass traces
# 3. find isotopic pairs
# 4. cosine filtering
# 5. peak picking
# 6. removing redundant envelopes

#%% files



mzmls=["C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/MNX-MeOH-neg-001.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/MNT-2TBA-001.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/MNT-TBA-001.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/MNX-UD-neg-001.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/MNX-neg-001.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/MNX-MeOH-pos-001.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/MNX-DMSO-neg-001.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/MNT-TBA-pos-001.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/MNX-DMSO-pos-1250-001.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/MNX-DMSO-neg-1250-001.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/MNT-2TBA-C18-neg-001.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/ACN-001.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/ACN-002.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/ACN-004.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/ACN-005.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/ACN-006.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/BM-001.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/DMIT-600pgul-001.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/DMIT-C18-neg-001.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/DMIT-neg-002.mzML",
"C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/DMIT-pos-001.mzML"]


#mzmls=["C:/MNX_analysis/Antwerp/LC_MS/240718/Data/mzml/MNT-2TBA-001.mzML"]

#%% Parameters

min_intensity=100
mass_blowup=10 #multiply with to convert to int datatype
isotope_window=np.arange(0,8+1)

trace_distance=3    #max mass difference post blowup for appending traces
max_distance=1      #max mass difference post blowup for finding isotope pairs
max_distance_filter=2      #max mass difference post blowup for filtering isotope peaks
cosine_window=5
min_cosine=0.8

output_folder="C://MNX_analysis//Antwerp//LC_MS//240718//Outputs"


#%% functions



def pick_all_peaks(ds,rel_height=0.5,extend_window=0,prominence=100,distance=10):

    Spectrum=ds.iloc[0,:]
    
    
    try:
        
        #identify periodicity
        p,_=find_peaks(Spectrum)
        periodicity=int(np.quantile(np.diff(p),0.05))
        savSpectrum=savgol_filter(pd.Series(Spectrum).fillna(0),2*periodicity,2)
    
    
        
        smoothing_window=periodicity
        if np.quantile(np.diff(p),0.01)>distance: #*2:
            sSpectrum=pd.Series(savSpectrum).fillna(0)
            
        while  np.quantile(np.diff(p),0.01)<=distance: #*2:
            #print("trying smoothing window: " +str(smoothing_window))
            smoothing_window+=periodicity
            sSpectrum=pd.Series(savSpectrum).fillna(0).rolling(window=smoothing_window).mean()  
            p,pi=find_peaks(sSpectrum,prominence=prominence)#,distance=distance)
    
    except:
        sSpectrum=Spectrum.copy()
        smoothing_window=0
        
    p,pi=find_peaks(sSpectrum,prominence=prominence)#,distance=distance)   
    pro,lb,rb=pi["prominences"],pi["left_bases"],pi["right_bases"]
    
    #s=np.argsort(pro)[::-1]
    #print("number of peaks: "+str(len(s)))
    
    ps=[]
    
    inflect_l=0
    for ix,_ in enumerate(p):
        #print(ix) #test
    
        w=max(p[ix]-lb[ix],rb[ix]-p[ix])
        l,r=p[ix]-w,p[ix]+w
        if l<0: l=0
        
        c=sSpectrum[l:r]
        x,y=c.index,c.values
        my=np.argwhere(x==p[ix])[0][0] #np.argmax(y)
        
        inflect_l=0
        if ix: 
            left_space=(p[ix]-p[ix-1])
            q=np.argwhere(np.diff(y)>=0)
            qs=q[(q<my) & (q>(my-left_space))]
            if len(qs):
                inflect_l=qs[np.argmin(y[qs])]
        
        dr=(y-y[my]*(1-rel_height))<0
        
        fwhm_l=0
        if sum(dr[:my]): fwhm_l=np.argwhere(dr[:my]).max()-extend_window
        fin_l=max(fwhm_l,inflect_l)
        
        inflect_r=my #len(x)-my #x[-1]-p[ix]
        if (ix+1)<len(p):
            right_space      =p[ix+1]-p[ix]
            q=np.argwhere(np.diff(y)<=0)
            qs=q[(q>my) & (q<(my+right_space))]
            if len(qs):
                inflect_r=qs[np.argmin(y[qs])]
           
        
        dr=(y-y[my]*(1-rel_height))<0
        fwhm_r=len(y)-1
        if sum(dr[my:])>extend_window: fwhm_r=my+np.argwhere(dr[my:]).min()+extend_window
        fin_r=min(fwhm_r,inflect_r)
                     
        ps.append([p[ix]-smoothing_window/2,    #peak
                   x[fin_l]-smoothing_window/2, #left_border
                   x[fin_r]-smoothing_window/2, #right border
                   max(y[fin_l:fin_r])])      #height
        
        
        # #Plot (testing)
        # fig,ax=plt.subplots()
        # plt.plot(c)
        # plt.scatter(p,sSpectrum[p],color="grey")
        # plt.scatter(x[my],y[my],color="red")
        # plt.vlines([x[inflect_l],x[inflect_r]],0,y[my],color="grey") #inflection point
        # plt.vlines([x[fwhm_l],x[fwhm_r]],0,y[my],color="green")      #fwhm
        # plt.vlines([x[fin_l],x[fin_r]],0,y[my],color="red")          #final boundaries
        # plt.xlim(x[fin_l]-100,x[fin_r]+100)
        # plt.ylim(0,y[my]*2)

    return np.array(ps).astype(int)

#%% 1. load mzml

if not os.path.exists("C:/MNX_analysis/Antwerp/LC_MS/240718/Outputs"):
    os.makedirs(output_folder)


for mzml in mzmls:
    # break #test
    fs=Path(mzml).stem
    
    print("reading mzML")
    print(mzml)
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(mzml, exp)
    
    ds=[]
    for ix,spec in enumerate(exp):
        
        m,i=spec.get_peaks()
        m=(m*mass_blowup).astype(np.int16)
        i=i.astype(np.uint32)
        
        p=find_peaks(i,prominence=10)[0]
        m,i=m[p],i[p]
        
        df=pd.DataFrame([[ix]*len(i),m,i],index=["scan","mass","intensity"]).T.astype(np.int32)
        ds.append(df)
    
    ds=pd.concat(ds)
    
    
    #%% 2. Extracting traces
    
    #array mapping
    um,us=np.unique(ds.mass),np.unique(ds.scan)
    mm,ms,=np.max(ds.mass),np.max(ds.scan)
    sm,ss=np.zeros(mm+1,dtype=np.uint16),np.zeros(ms+1,dtype=np.uint16)
    sm[um]=np.arange(len(um))
    ss[us]=np.arange(len(us))
    
    groups=ds.groupby("scan")
    
    
    
    mass_traces,intensity_traces=np.zeros([len(um),len(us)],dtype=np.uint16),np.zeros([len(um),len(us)],dtype=np.uint32)
    
    # ma=[]
    for n,g in groups:
        
        m,i=g.mass.values,g.intensity.values
        
        if n:
            
            m1=mass_traces[:,n-1]
            
            pairs=np.argwhere(abs(m1.reshape(-1,1)-m)<=trace_distance)
            pm=np.vstack([m1[pairs[:,0]],m[pairs[:,1]]]).T #old, new
            nopairs=~np.in1d(m,pm[:,1])
    
            mass_traces     [pairs[:,0],n]=m[pairs[:,1]]
            intensity_traces[pairs[:,0],n]=i[pairs[:,1]]
            
            mass_traces     [sm[m[nopairs]],n]=m[nopairs]
            intensity_traces[sm[m[nopairs]],n]=i[nopairs]
            
    
        else:    
            mass_traces[sm[m],n]=m
            intensity_traces[sm[m],n]=i
    
    
    #filter on minimum intensity
    q=intensity_traces.max(axis=1)>min_intensity
    mass_traces,intensity_traces,um=mass_traces[q],intensity_traces[q],um[q]
    mean_masses=(pd.DataFrame(mass_traces).replace(0,np.nan).mean(axis=1)/mass_blowup).round(2).values
    
    #plot traces
    idf=pd.DataFrame(intensity_traces,index=mean_masses).sort_index()
    mdf=pd.DataFrame(mass_traces,index=mean_masses).sort_index()

    fig,ax=plt.subplots()
    sns.heatmap(idf,cmap="Greens",robust=True) #vmax=np.percentile(idf,99))#,cmap="coolwarm")#,robust=True)
    plt.title(fs)
    plt.savefig(str(Path(output_folder,fs+"_traces.png")),dpi=300)
    #save tracemap
    
    #%% 3. find isotopic pairs
    
    #subtract ints
    i=mdf.index.astype(np.int16).values
    di=i-i.reshape(-1,1)
    
    #subtract decimals
    d=((mdf.index-i)*10).astype(np.int8).values
    dd=abs(d-d.reshape(-1,1))
    
    pairs=np.argwhere((di >= np.min(isotope_window)) & (di <= np.max(isotope_window)) & (dd<=max_distance)) # di in isotopic window &  dd < cutoff
    
    #%% 4. cosine filtering (and merging isotopes)
    
    #https://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat
    #https://stackoverflow.com/questions/34070278/vectorized-cosine-similarity-calculation-in-python
    
    
    #test=idf.reset_index()
    
    envelopes=[]
    retention_times=[]
    
    spairs=np.array_split(pairs,np.argwhere(np.diff(pairs[:,0])>0)[:,0]+1)
    inc=0
    for ix,p in enumerate(spairs):
    
        #print(ix)
    
#         if ix==4900: #465: #test 
#             break
                
        d=idf.iloc[p[:,1],:]
        m=mdf.iloc[p[:,1],:]

        #calculate cosine
        ints=d.values
        
        r,c=ints.shape
        res=np.zeros([r-1,c-cosine_window])
        
        mx=np.argwhere(p[0,0]==p[:,1]).flatten() 
        mxi=np.argwhere(p[0,0]!=p[:,1]).flatten() 
        
        with np.errstate(invalid='ignore'):
        
            for i in range(ints.shape[1]-cosine_window):
            
                s=ints[:,i:i+cosine_window]
                A=s[mx,:] #this is wrong, and should be linked to the main mass
                B=s[mxi,:]
                nA=norm(A)
                
                if nA: 
                    res[:,i]= np.dot(A/nA,(B/norm(B,axis=1)[...,None]).T)
    
        
        res=np.nan_to_num(res)
        #sns.heatmap(ints) #test
        #sns.heatmap(res) #test
        
        #merge isotopes
        q=np.argwhere(res<min_cosine)
        
        #not sure if this is correct
        d.values[mxi[q[:,0]],q[:,1]]=0
        m.values[mxi[q[:,0]],q[:,1]]=0
        m.index=m.replace(0,np.nan).mean(axis=1)/mass_blowup
        d.index=m.index
        
        #filter again on mass differences
        d0=d.index[mx].values
        q=abs(d.index-d.index.fillna(0).astype(int)-d0+d0.astype(int))<max_distance/mass_blowup
        d,m=d[q],m[q]

        di=np.round(d.index-d0,0).astype(int)
        ds,ms=d.set_index(di),m.set_index(di)
        ds=ds.groupby(ds.index).max() #summing would be wrong because of trace duplication
        ms=ms.replace(0,np.nan).groupby(ms.index).mean() 
        ds.index=(ms.mean(axis=1)/mass_blowup).values.round(2)
        
        # #%%test
        # fig,ax=plt.subplots()
        # for i in ds.values:
        #     plt.plot(i)
        # plt.legend(ds.index)
        
        #5. peak picking (is this even needed?)
        #split traces
        ps=pick_all_peaks(ds)
        for i in ps:
    
  
            su=ds.iloc[:,i[1]:i[2]]
            su=su[su.sum(axis=1)>0]
            
            if len(su)<2:
                continue
            
            ### final filtering ###
            
            #mass filtering
            si=(su.index-su.index.astype(int))*mass_blowup
            su=su[abs(si-si[0])<max_distance_filter] # max_distance] #or do this ppm based???!
        
            #cosine filtering
            A=su.iloc[0,:].values
            B=su.iloc[1:,:].values
            
            with np.errstate(invalid='ignore'):
    
                cos=np.hstack([[1],np.dot(A/norm(A),(B/norm(B,axis=1)[...,None]).T)])
            su=su[cos>=min_cosine]
        
            #remove gapped #2 (could be too harsh filtering)
            g=np.argwhere(np.diff(su.index)>1.5)
            if len(g):
                su=su.iloc[:g[0][0]+1,:]
                
            if len(su)<2:
                continue
        
            else:
    
                d=su.sum(axis=1).reset_index()
                d.columns=["mass","intensity"]
                d["ix"]=inc
                d["isotope"]=(d.mass-d.mass[0]).round(0).astype(int)
                envelopes.append(d)
            
                retention_times.append(pd.DataFrame(np.hstack([i[0:3],[inc]]),index=["peak","start","end","ix"]).T)
                
                
                # if inc==3354:
                #     ""+1
                
                inc+=1
                print(inc)
                
                

                #test
                # fig,ax=plt.subplots()
                # sns.heatmap(su,robust=True)
                # plt.title(ix)
  
    
    envelopes=pd.concat(envelopes).set_index("ix") 
    retention_times=pd.concat(retention_times).set_index("ix") 
    mono=envelopes.groupby("ix",sort=False).nth(0).mass
    
    
    
    #%% 6. remove redundant envelopes
    
    
    
    exclusion=set()
    signatures=[]
    groups=envelopes.groupby("mass")
    for n,g in groups:
        
    
        indices=envelopes.reset_index().set_index("mass").loc[envelopes.loc[g.index].mass].ix.drop_duplicates().values #redundant evelopes
        indices=indices[~np.in1d(indices,exclusion)]
    
        if len(indices):    
            exp=envelopes.loc[indices].reset_index()
    
         
            exp=exp.sort_values(by=["isotope","intensity"],ascending=False)
            f=exp[exp["ix"]==exp.iloc[0]["ix"]]
            exp["nmass"]=(exp.mass-f.loc[f["isotope"]==0,"mass"].iloc[0]).round(0).astype(int)
            
            #groupby 
            
            pv=exp.pivot(columns="nmass",index="ix",values="intensity").drop_duplicates() 
            npv=pv.divide(pv.sum(axis=1),axis=0)
            
            nu=exp[~exp.ix.isin(pv.index)].index.values.tolist() #add duplicates to "ix"-clusion list  
            if len(nu): exclusion.update(nu)   
         
            #for reach row combine overlapping mass/isotope patterns
    
            c=npv.iloc[0]
            q=c.notnull()
            
            ex=[]
            for ir,r in npv.iloc[1:].iterrows():
                q1=q & r.notnull()
                A,B=r[q1],c[q1]
                
                if min_cosine<np.dot(A/norm(A),B/norm(B)):
                    #merge missing isotopes?
                    ex.append(ir) 
                
            
            exclusion.update([c.name]+ex)
            signatures.append(c.name)
    
    final_envelopes=envelopes.loc[np.unique(signatures)]
    fpv=final_envelopes.pivot(values="intensity",columns="isotope")
    
    
    fpv["mass"]=mono.loc[fpv.index]
    fpv[["fwhm_peak","fwhm_start","fwhm_end"]]=retention_times.loc[fpv.index]
    
    #save isotopic envelopes
    fpv.to_csv(str(Path(output_folder,fs+"_isotope_envelopes.tsv")),sep="\t")

    #save mass traces
    idf.to_csv(str(Path(output_folder,fs+"_intensity_traces.tsv")),sep="\t")
    mdf.to_csv(str(Path(output_folder,fs+"_mass_traces.tsv")),sep="\t")
