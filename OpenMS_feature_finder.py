# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:44:58 2024

@author: hkleikamp
"""


file="C:/Proteomics_Jeanine/mzML/B004241_Bp_PRC-2296_Remaut_shotgun_1_20180722184121.mzML"


#%%
import pyopenms
import pandas as pd
import numpy as np
from pathlib import Path
#%%



#load ms1
exp = pyopenms.MSExperiment()
pyopenms.MzMLFile().load(file, exp)
exp.setSpectra([s for s in exp.getSpectra() if s.getMSLevel()==1])  

#centroid spectra  
with open(file,"r") as f:
    lines=f.readlines(10000)
if "centroid" not in "".join(lines):
    print("centroiding")
    exp_centroid = pyopenms.MSExperiment()
    pyopenms.PeakPickerHiRes().pickExperiment(exp, exp_centroid) 
    exp=exp_centroid
exp.sortSpectra(True)



#detect mass traces
mass_traces = []
mtd = pyopenms.MassTraceDetection()
mtd_params=mtd.getParameters() #these parameters could still need some tweaking
mtd_params.setValue("mass_error_ppm", 20.0) # set according to your instrument mass error
mtd_params.setValue("noise_threshold_int", 1000.0) # adjust to noise level in your data
mtd_params.setValue('min_sample_rate', 0.66) 
mtd_params.setValue('min_trace_length', 0.3)
mtd.setParameters(mtd_params)
mtd.run(exp, mass_traces, 0)


    
    

#split mass traces
mass_traces_split = []
mass_traces_final = []
epd = pyopenms.ElutionPeakDetection()
epd_params = epd.getDefaults()
epd_params.setValue("width_filtering", "auto") 
epd.setParameters(epd_params)
epd.detectPeaks(mass_traces, mass_traces_split)
if epd.getParameters().getValue("width_filtering") == "auto":
    epd.filterByPeakWidth(mass_traces_split, mass_traces_final)
else:
    mass_traces_final = mass_traces_split

#link to features
fm = pyopenms.FeatureMap()
feat_chrom = []
ffm = pyopenms.FeatureFindingMetabo()
ffm_params = ffm.getDefaults() #still might need some tweaking
ffm_params.setValue("isotope_filtering_model", "none")
ffm_params.setValue('charge_upper_bound', 6)
ffm_params.setValue("mz_scoring_by_elements", "false")
ffm_params.setValue("report_convex_hulls", "false")
ffm_params.setValue("report_summed_ints", "true")
ffm_params.setValue("enable_RT_filtering", "false")
ffm.setParameters(ffm_params)
ffm.run(mass_traces_final, fm, feat_chrom)
fm.setUniqueIds()
fm.setPrimaryMSRunPath([file.encode()])

pyopenms.FeatureXMLFile().store(str(Path(Path(file).parents[0],"pepiso_"+Path(file).stem+'.featureXML')), fm)


#%% inspect features to store as tabular format

#get MZ get Charge get RT
fdf=pd.DataFrame([[i.getMZ(), i.getCharge(), i.getRT()] for i in fm],columns=['monoisoMz', 'charge', 'rtApex'])


fdf.to_csv(file.replace(Path(file).suffix,'_openms_features.tsv'),sep="\t")

