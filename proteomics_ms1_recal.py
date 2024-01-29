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

from bisect import bisect_left #Fast pick close
from scipy.signal import find_peaks
import more_itertools as mit   #make windows
from scipy.optimize import curve_fit #curve fit
#%% Inputs


PSMs_file="C:/Proteomics_Jeanine/MSFragger_open_single_strain/B004241_Bp_PRC-2296_Remaut_shotgun_1_20180722184121.pin" #pin file or any tabular format
mzML_file='C:/Metabolomics/transfer_290523_files_37e684c1/mzml/B004241_Bp_PRC-2296_Remaut_shotgun_1_20180722184121.mzML'

#utils files
unimod_file=str(Path(basedir,"utils","unimod_parsed.txt"))                       # elemental compositions of modifications
isotope_file=str(Path(basedir,"utils","natural_isotope_abundances.tsv"))         # nist natural isotope abundances
amino_acids_file=str(Path(basedir,"utils","AA_formulas.txt"))                    # elemental compositions of amino acids
calibrants_file=str(Path(basedir,"utils","calibrants.tsv")) # internal standards used for calibration (DOI: 10.1016/j.aca.2008.04.043)


#%% params

#reading peptide file
FDR_cutoff=0.05
score_column="hyperscore"
decoy_delimiter="rev_"

#feature extraction
neighbouring_scans=10
ppm=20        

#windowing
batch=10
overlap=5
min_count=2

#fitting
fit_method="monod" #monod for orbitrap, linear for TOF
minimum_rsquared=0.5
minimum_calibrants=5
smoothing_window=3

#%% fixed variables

charge_mass    =1.007276466621     #proton_mass (or put electron mass here in case of negative charge)
isotope_mass   =1.002881202641     #1.0033548350722992=mass shift of C13-C12 (this actually depends on the chemical composition!!! for example isotopes H2-1: 1.00627, N15-14: 0.99699)

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
 

def take_closest(myList, myNumber):
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def lin(x,a,b):
    return a*x+b

def monod(x,vmax,ks,b):
    return vmax*x/(ks+x)+b
    
def r2(ydata,fitted_data):
    ss_res = np.sum((ydata- fitted_data)**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    return  1 - (ss_res / ss_tot)


def calibrate(data,method,p0="",Plot=False,save=False,fev=10000,bounds=(-np.inf, np.inf)):
    
    try:
    
        x,y=data["mz"],data["ppm_diff"]
    
        if method=="linear":
            if len(p0):
                popt, _ = curve_fit(lin,x,y,p0=p0,maxfev=fev,bounds=bounds)
            else:
                popt, _ = curve_fit(lin,x,y,maxfev=fev,bounds=bounds)
            f=lin(x,*popt)            
                
        if method=="monod":
            if len(p0):
                popt, _ = curve_fit(monod,x,y,p0=p0,maxfev=fev,bounds=bounds)
            else:
                popt, _ = curve_fit(monod,x,y,maxfev=fev,bounds=bounds)
            f=monod(x,*popt)
        
        rs=r2(y,f)
            
        if Plot:
           fig,ax=plt.subplots()
           plt.scatter(x,y,s=1,label="PSMs")
           plt.plot(x,f,label=method+" fit r2: "+str(round(rs,3)),c="r")
           plt.ylabel("ppm mass shift")
           plt.xlabel("m/z")
           plt.title("global calibration")
           plt.legend(loc='best',markerscale=3)
           
           if save: #extra toggle for testing
               fig.savefig(mzML_file.replace(".mzML","_global_fit.png"),dpi=300)
            
        return rs, popt 
    except Exception as error:
        print("An error occurred:", type(error).__name__, "–", error)
        return 0,[]

def isnumber(x):
    return isinstance(x, (int, float, complex)) and not isinstance(x, bool)


def correct_mass(m,  #masses
                 f): #fit data

    tog=0
    if isnumber(m): 
        m=[m]
        tog=1
        
    par=f[param_cols].tolist()
    p=np.zeros(len(m))
    m0=m-d0
    ql,qu=(m0<f.mass_low),(m0>f.mass_high)
    qf=(~ql & ~qu)
    p[ql]=f.ppm_low
    p[qu]=f.ppm_high
    
    if fit_method=="monod":
        p[qf]=monod(m0[qf],*par)
    if fit_method=="linear":
        p[qf]=lin(m0[qf],*par)
        
    r=m*(1+p/1000000)
    #r=m*(1-p/1000000)

    if tog:
        return r[0]
        
    return r


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
isotope_table["mass_number"]=isotope_table["mass_number"].astype(int)
isotope_mass=isotope_table.loc[isotope_table["symbol"]=="C","Relative Atomic Mass"].diff().values[-1]

#parse calibrants mass
cs=calibrants["Mz"].drop_duplicates().sort_values()

#%% 1. Read mzML



print("reading mzML")
exp = pyopenms.MSExperiment()
pyopenms.MzMLFile().load(mzML_file, exp)


ms1=[s for s in exp.getSpectra() if s.getMSLevel()==1]
ms1_scans=np.array([spec.getNativeID().split("scan=")[1] for spec in ms1]).astype(int)
ms1_peaks=[i.get_peaks() for i in ms1]
ms1_masses=[i[0] for i in ms1_peaks]

ms2=[s for s in exp.getSpectra() if s.getMSLevel()==2]
ms2_peaks=[i.get_peaks() for i in ms2]
ms2_masses=[i[0] for i in ms2_peaks]

ms2_info=pd.DataFrame([[int(spec.getNativeID().split("scan=")[1]),
                        spec.getPrecursors()[0].getCharge(),
                        spec.getPrecursors()[0].getMZ()] for spec in ms2],columns=["scan","Charge","Precursor_mz"])

ms2_scans=ms2_info.scan.values

#%% 2. Read PSMs file

#1a read PSMs file (pin file)
psms=read_unknown_delim(PSMs_file)  #tsv file with required columns: scan, Peptide 
if PSMs_file.endswith(".pin"):      #pin specfic prep
    psms["scan"]=psms["ScanNr"].astype(int)
    psms["Peptide"]=psms["Peptide"].str.split(".").apply(lambda x: ".".join(x[1:-1]))


# 1b calculate chemical composition
psms=EleCounter(psms)

#calculate psm diff
psms=psms.merge(ms2_info,on="scan",how="left")
psms["mz"]=(psms.mass+charge_mass*psms.Charge.values)/psms.Charge.values
psms["ppm_diff"]=(psms["mz"]-psms["Precursor_mz"])/psms["mz"].values*1000000
psms=psms[abs(psms["ppm_diff"])<ppm]


#%% 3. Find neighbouring scans
peptide_scans=psms["scan"].astype(int) #.sort_values()
inc,c,x=0,0,0
neighbours,ds,m1s,uscans=[],[],[],[]

for m2 in peptide_scans:
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

epsms=psms.copy()
epsms["scan"]=neighbours
epsms=epsms[["ppm_diff","scan","mz"]].explode("scan").drop_duplicates()



#%% 4. extract calibrants
print("extracting background calibrants")
ds=[]
for ix,m in enumerate(ms1_masses):

    closest=np.array([take_closest(m,i) for i in cs])
    dppm   =abs(closest-cs)/cs*1000000 #ppm error 
    q=dppm<=ppm   #retain those within ppm tol
    t,f=cs[q],closest[q]
    
    d=pd.DataFrame([t,(t-f)/f*1000000]).T
    d.columns=["mass",ms1_scans[ix]]
    d=d.set_index("mass")
    ds.append(d)

ds=pd.concat(ds,axis=1)

#Inter quartile range 2
means=ds.mean(axis=1).sort_index()
q1,q3=np.percentile(means,25),np.percentile(means,75)
fmeans=means[(means<q3+1.5*(q3-q1)) & (means>q3-1.5*(q3-q1))]

#peak picking
peaks="start" 
while len(peaks): 
    peaks, _=find_peaks(fmeans,prominence=(q3-q1)/2) #not sure why q3-q1 but it looked good for this dataset
    fmeans=fmeans[~fmeans.index.isin(fmeans.index[peaks])]

ds.columns=ms1_scans
ds=ds.loc[fmeans.index].unstack().reset_index().dropna()
ds.columns=["scan","mz","ppm_diff"]



#%% Calibration


# Global calibration
print("Global calibration")
ca=pd.concat([ds,epsms]).sort_values(by=["scan","mz"]).set_index("scan")
windows = list(mit.windowed(np.unique(ca.index), n=batch, step=batch-overlap))
wdf=pd.DataFrame(np.hstack(windows),columns=["scan"])
wdf["window_index"]=np.repeat(np.arange(len(windows)),batch)
wdf=wdf[wdf.scan.notnull()]

wdf=wdf.merge(ca,on="scan",how="left").drop_duplicates()
s=wdf.groupby(["window_index","mz"]).size()
wdf=wdf.set_index(["window_index","mz"]).loc[s[s>+min_count].index].reset_index()
t=wdf.groupby(["window_index","mz"],sort=False)[["ppm_diff"]].mean().reset_index().set_index("window_index").sort_values(by="mz")

d=fmeans.reset_index()
d.columns=["mz","ppm_diff"]
d0=d.mz[0]

t.mz=t.mz-d0
d.mz=d.mz-d0

#lin fit
rs,popt=calibrate(d,method="linear",Plot=True,save=True)

if fit_method=="monod":
    p0=[lin(d.mz.values[-1],*popt),d.mz.values[-1]/2,0]
    rs,popt=calibrate(d,method=fit_method,p0=p0,Plot=True,save=True)

param_cols=["param_"+str(i) for i in range(len(popt))]
    
plt.gca()
plt.ylim(-2,2)
print("fitting method : "+fit_method)
print("r_squared : "+str(rs))
print("base parameters : "+str(popt))

#make bounds 
bounds_scale_factor=2
bounds_add_factor=5
e=np.array([popt*bounds_scale_factor,popt/bounds_scale_factor]).T
lower_bounds,upper_bounds=e.min(axis=1),e.max(axis=1) 
lower_bounds=lower_bounds-bounds_add_factor
upper_bounds=upper_bounds+bounds_add_factor

# Local calibration
print("Local calibration")

t=t.sort_values(by=["window_index","mz"])
ms=t.groupby("window_index",sort=False)
cal=[]
for ix,g in ms:
    
    if ix%500==0:
        
        print("fraction completed: "+str(round(ix/len(windows),3)))
    
    
    s=windows[ix]
    if None in s: s=[i for i in s if i!=None]
    d=g.set_index("mz").squeeze()

    
    q1,q3=np.percentile(d,25),np.percentile(d,75)
    d=d[(d<q3+1.5*(q3-q1)) & (d>q3-1.5*(q3-q1))]
    if not len(d): continue
    
    peaks="start" 
    while len(peaks): 
        peaks, _=find_peaks(d,prominence=(q3-q1)/2) #not sure why q3-q1 but it looked good for this dataset
        d=d[~d.index.isin(d.index[peaks])]
   
    if len(d)>len(popt):
        r,p=calibrate(d.reset_index(),method=fit_method,p0=popt,bounds=(lower_bounds,upper_bounds),fev=100)#,Plot=True)
        
        #store border values
        v0,v1=d.index[0],d.index[-1]
        if fit_method=="linear":
            b0,b1=lin(v0,*popt),lin(v1,*popt)
        if fit_method=="monod":
            b0,b1=monod(v0,*popt),monod(v1,*popt)
        
        cal.append([ix,
                    s[0],s[-1], # scan borders
                    v0,v1,      # mass borders
                    b0,b1,      # ppm for mass borders 
                    r,len(d)]+p.tolist())
        


cal=pd.DataFrame(cal,columns=["window",
                              "start_scan","end_scan",
                              "mass_low","mass_high",
                              "ppm_low","ppm_high",
                              "r-squared","number of calibrants"]+param_cols)  


cal=cal[cal["r-squared"]>minimum_rsquared]
cal=cal[cal["number of calibrants"]>minimum_calibrants]
cal[["ppm_high","ppm_low"]+param_cols]=cal[["ppm_high","ppm_low"]+param_cols].rolling(smoothing_window).median().bfill()
cal["start_scan"][1:]=cal["end_scan"][0:-1].values+1 #remove overlapping window and forward fill

#plotting
for i in param_cols:
    fig,ax=plt.subplots()
    plt.scatter(cal["start_scan"],cal[i],s=0.1)
    plt.xlabel("scan")
    plt.ylabel(fit_method+" "+i)
    plt.title("local calibration")
    fig.savefig(mzML_file.replace(".mzML","_"+fit_method+" "+i+".png"),dpi=300)

#add missing scans
cal["scan"]=cal.apply(lambda x: np.arange(x["start_scan"],x["end_scan"]+1),axis=1)
ecal=cal[["scan","mass_low","mass_high","ppm_low","ppm_high"]+param_cols].explode("scan")
ecal=pd.DataFrame(np.hstack([ms1_scans,ms2_scans]),columns=["scan"]).sort_values(by="scan").merge(ecal,on="scan",how="left").ffill().set_index("scan")



#%% Correcting masses

print("Correcting masses")


rs=[]
c_exp=exp.__copy__()
specs=c_exp.getSpectra()

for ix,spec in enumerate(c_exp):
    
    #get data
    ms=spec.getMSLevel()
    scan=int(spec.getNativeID().split("scan=")[1])
    peaks=spec.get_peaks()
    m=peaks[0]
    
    #fit spectrum
    f=ecal.loc[scan]
    c=correct_mass(m,f)
    
    specs[ix].set_peaks([c,peaks[1]])
    
    #fit precursor
    if ms==2:
        precs=spec.getPrecursors()
        for ip,prec in enumerate(precs):
            r=correct_mass(prec.getMZ(),f)
            precs[ip].setMZ(r)
            rs.append([scan,r])
        specs[ix].setPrecursors(precs)
        
c_exp.setSpectra(specs)
pyopenms.MzMLFile().store(mzML_file.replace(".mzML","_calibrated.mzML"), c_exp)

#%% Compare precursors

psms=psms.merge(pd.DataFrame(rs,columns=["scan","corrected_Mz"]),on="scan",how="left")
psms["calibrated_ppm_diff"]=(psms["mz"]-psms["corrected_Mz"])/psms["mz"]*1000000

mb="uncalibrated, median ppm : "+str(round(psms["ppm_diff"].median(),2))
ma="calibrated,   median ppm : "+str(round(psms["calibrated_ppm_diff"].median(),2))
print(mb)
print(ma)

fig,ax=plt.subplots()
psms["ppm_diff"].plot.hist(bins=100,color=(0.5, 0, 0, 0.3))
psms["calibrated_ppm_diff"].plot.hist(bins=100,color=(0, 0.5, 0, 0.3))
plt.gca()
plt.legend([mb,ma])
plt.title("ppm after calibration")
plt.xlabel("ppm")
plt.ylabel("psms mass deviation")

fig.savefig(mzML_file.replace(".mzML","_calibration_histogram.png"),dpi=300)


