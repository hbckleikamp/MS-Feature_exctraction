# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:33:13 2024

@author: hkleikamp
"""
from msbuddy import Msbuddy, MsbuddyConfig
from msbuddy.base import MetaFeature, Spectrum
import numpy as np
import pandas as pd


#parallel setup for windows (execute from command line)

#%%

fdf=pd.read_csv("C:/Proteomics_Jeanine/mzML/B004241_Bp_PRC-2296_Remaut_shotgun_1_20180722184121OpenMS_isotopologue.tsv",index_col=[0],sep="\t") #.iloc[:1000]


fdf["id"]=np.arange(len(fdf))
icols=[i for i in fdf.columns if i.endswith("_isotope")]
mcols=[i for i in fdf.columns if i.endswith("_mass")]


charge_mass    =1.007276466621     #proton_mass (or put electron mass here in case of negative charge)
#Buddy also needs charge correction!
#If it should be mass, Then dont call this mz you idiot!!!!!
fdf[mcols]=fdf[mcols]*fdf[["charge"]].values-charge_mass*fdf[["charge"]].values+charge_mass



#%%
features=[]

c=0
for _,r in fdf.iterrows():
   

    ms1_spec = Spectrum(mz_array = r[mcols].dropna().values,
                        int_array = r[icols].dropna().values)

    features.append( MetaFeature(identifier = r.id,
                              mz = r["0_mass"], #r.monoisoMz, 
                              charge = r.charge,
                              adduct = '[M+H]+',
                              ms1 = ms1_spec))
    
    
    c+=1




#%%



#parallel processing doesnt work directly from script
# create a MsbuddyConfig object
msb_config = MsbuddyConfig(ms_instr="orbitrap", # supported: "qtof", "orbitrap", "fticr" or None
                                                # custom MS1 and MS2 tolerance will be used if None
                                                # highly recommended to fill in the instrument type
                           ppm=True,  # use ppm for mass tolerance
                           ms1_tol=5,  # MS1 tolerance in ppm or Da
                           ms2_tol=10,  # MS2 tolerance in ppm or Da
                           p_range=[0,3], #phosphate groups tend to get lost in MS  
                           batch_size=10000,
                           s_range=[2,15], #default is [0,15] #set this in case you want to force the annotation of dithiolenes
                            parallel=True, # enable parallel processing, see note below
                            n_cpu=5, # number of CPUs to use
                           
                           halogen=False,
                           timeout_secs=200)

# instantiate a Msbuddy object with the parameter set
engine = Msbuddy(msb_config)




# add to the Msbuddy object, List[MetaFeature] is accepted
engine.add_data(features)

if __name__== '__main__':

    engine.annotate_formula()
    
    #How to constrain elemental composition
    
    #%%
    
    d=engine.data
    
    dfs=[]
    for i in d:
        rs=i.candidate_formula_list
        for r in rs:
            
            f=r.formula
        
            df=pd.DataFrame([[i.identifier,
                              i.mz,
                             f.mass,
                             r.estimated_fdr,
                             r.estimated_prob,
                             r.ml_a_prob,
                             r.ms1_isotope_similarity,
                             r.normed_estimated_prob]+f.array.tolist()], 
                             
                             columns=["id",
                                      "measured_mass",
                                      "buddy_mass",
                                      'estimated_fdr',
                                      'estimated_prob',
                                      'ml_a_prob',
                                      'ms1_isotope_similarity',
                                      'normed_estimated_prob']+f.alphabet )   
      
            dfs.append(df)
        
    dfs=pd.concat(dfs)
    
    
    [dfs.pop(i) for i in ['Na', 'Br', 'Cl', 'F', 'I', 'K'] if i in dfs.columns]
    
    dfs.to_csv("C:\Metabolomics\Buddy\Buddy_python_OpenMS_charge_corrected.tsv",sep="\t")
    print("done")
    #%% postfiltering can be done after
    #rdbe 
    #isotopic envelope
    #do random forest to decide right and wrong hits?