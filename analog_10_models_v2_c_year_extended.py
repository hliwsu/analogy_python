%reset -f
import numpy as np
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
# Starting time
starting_clock = datetime.now()  
print(starting_clock)
# creating time and durations
t_concurrent = t0 = datetime.strptime("1895-01", '%Y-%m').date()
t1 = datetime.strptime("2099-12", "%Y-%m").date()
t_total=[t_concurrent.strftime("%Y-%m")]
while t_concurrent < t1:
    t_concurrent =t_concurrent + relativedelta(months=1)
    t=datetime.strftime(t_concurrent, "%Y-%m")
    t_total.append(t)
dist_na=np.load("C:/Users/haoli/OneDrive/Work/Analogy/Python codes/sel_data/error_0.npy")
dist_id_all=list(range(0,503))
dist_id_sel=[e for e in dist_id_all if e not in dist_na]
dist_id_sel=list(dist_id_sel)
dist_names_valid=np.load("C:/Users/haoli/OneDrive/Work/Analogy/Python codes/analog/dist_n_a.npy")
dist_names_all=np.load("C:/Users/haoli/OneDrive/Work/Analogy/Python codes/analog/dist_n_all.npy")
dist_names_id=np.load('C:/Users/haoli/OneDrive/Work/Analogy/Python codes/analog/dist_id_valid.npy')

"""
Note 1: 
data range
prism results time dimension: 1895-01-16 00:00:00 ... 1950-12-16 00:00:00 upper, 
                              1951-01-16 00:00:00 ... 2014-12-16 00:00:00 lower.
MACA rcp climate results time dimension:  2006-01-15 2006-02-15 ... 2099-12-15

NPP, Vtype MC2 historical: 1895-2014, annually
NPP, Vtype MC2 future: 2015-2099, annually
Note 2: 
notations:
    _h: historical
    _p1: part one, PRISM model results    

Note 3: 
matrix check
arr = np.arange(704160).reshape(1440, 489)
# yr, mon, districts 
am=arr.reshape((120,12,489)).mean(axis=(1))
# correct formation

Note 4: 
Conformable matrices

Note A1: for my own records
index matrices (test purposes)
years, year, yr
yr_A=yr[(1975-1895):(2015-1895)]
yr_B=yr[-50:,]
"""
# ***obtain folder and file names for the following loops***
# define models**
model=["rcp45_cgcm3", "rcp45_cm5", "rcp45_es365", "rcp45_m", "rcp45_resm1", "rcp85_cgcm3", "rcp85_cm5", "rcp85_es365", "rcp85_mr", "resm1"]
## file_suffix=".npy"
## model_file_suffix=[m+file_suffix for m in model ]
f_d=f_n=f_n_ext=[]
f = open('results', 'a')
# obtain paths for future and historical data
for model_file in model:
    locals()["future_n_p_tma_tmi_v_"+model_file]=[]
    ppt_future=[];tmax_future=[];tmin_future=[] # not using a=b=[] in here as we only need to keep the last run in the loops
    mc2_hist=[]; prism_1_hist=[]; prism_2_hist=[] # not using a=b=[] in here as we only need to keep the last run in the loops
    for dirpath, dirnames, filenames in os.walk("C:/Users/haoli/OneDrive/Work/Analogy/Python codes/data_filtered_june16_june20"):
        # for filename in [f for f in filenames if f.endswith("rcp45_cgcm3.npy")]:
        for filename in [f for f in filenames if f.endswith(model_file+".npy")]:
            locals()["future_n_p_tma_tmi_v_"+model_file].append(os.path.join(dirpath, filename))
        for filename in [f for f in filenames if f.endswith("mc2_hist.npy")]:
            mc2_hist.append(os.path.join(dirpath, filename)) # only keep the last loop
        for filename in [f for f in filenames if f.endswith("1950_PRISM.npy")]:
            prism_1_hist.append(os.path.join(dirpath, filename)) # only keep the last loop
        for filename in [f for f in filenames if f.endswith("2014_PRISM.npy")]:
            prism_2_hist.append(os.path.join(dirpath, filename)) # only keep the last loop  
        for filename in [f for f in filenames if f.startswith("ppt_rcp")]:
            ppt_future.append(os.path.join(dirpath, filename)) # only keep the last loop  
        for filename in [f for f in filenames if f.startswith("tmax_rcp")]:
            tmax_future.append(os.path.join(dirpath, filename)) # only keep the last loop  
        for filename in [f for f in filenames if f.startswith("tmin_rcp")]:
            tmin_future.append(os.path.join(dirpath, filename)) # only keep the last loop  
j = 0
# index array: years
yr=np.arange(1895,2100)
# Processing PPT, Tmax, and Tmin, and creating matrices A (historical), B (analog pool), and C (reference, pre-correction of PCA, time series).
A=[];B=[];C=[]
distances=np.zeros(shape=(len(dist_id_sel),len(dist_id_sel)))
# min_dist=np.zeros(shape=(len(dist_id_sel),len(dist_id_sel)))
m_control=0 # set model control index
for mdl in model:
    i=0 # set item control index
    # loop for ppt, tmax, tmin that has MACA historical inputs with 10 simulation models
    locals()['A_'+mdl]=[];locals()['B_'+mdl]=[] 
    for item in ["ppt", "tmax", "tmin"]:
        locals()[item+'_1895_1950']=np.load(prism_1_hist[i]) # load historical data (PRISM series)
        locals()[item+'_1951_2014']=np.load(prism_2_hist[i]) # load historical data (PRISM series)
        locals()[item+'_h']=np.concatenate((locals()[item+'_1895_1950'],locals()[item+'_1951_2014'])) # get the historical data set
        locals()[item+'_yr_mean_p1']=locals()[item+'_h'].reshape((int(locals()[item+'_h'].shape[0]/12),12,len(dist_id_sel))).mean(axis=(1)) # reshape historical data set according to years
        yr_p1=len(locals()[item+'_yr_mean_p1'])+1894 # obtain a year index
        # obtain future data
        locals()[item+"_"+mdl]=np.load(ppt_future[m_control]) # load future data (MC2 results series). It can also be obtained using "future_n_p_tma_tmi_v_" index but I decided to save some spaces:)  
        locals()[item+'_yr_mean_p2']=locals()[item+"_"+mdl].reshape((int(locals()[item+"_"+mdl].shape[0]/12),12,len(dist_id_sel))).mean(axis=(1)) # reshape future data set according to years
        locals()[item+'_yr_mean_p2']=locals()[item+'_yr_mean_p2'][-(2099-yr_p1):,:] # keeping PRISM data in the duplicate years while abondaning future data in conflict years
        locals()[item+'_yr_mean']=np.concatenate((locals()[item+'_yr_mean_p1'], locals()[item+'_yr_mean_p2']), axis=0) # generate full year datasets, 1895-2099, 205 years total
        ## creating sub A matrix (part 1/2: ppt, tmax, tmin, historical matrix, 1975-2015)
        locals()[item+'_A']=locals()[item+'_yr_mean'][(1975-1895):(2015-1895),].mean(axis=0)  #  yr_A=yr[(1975-1895):(2015-1895)] ## check years
        locals()[item+'_'+mdl+'_A_p1']=np.append(A,locals()[item+'_A'])
        ## creating sub B matrix (part 1/2: ppt, tmax, tmin, future matrix, 2050-2099 )
        locals()[item+'_B']=locals()[item+'_yr_mean'][2050-2099+1:,].mean(axis=0)
        locals()[item+'_'+mdl+'_B_p1']=np.append(B,locals()[item+'_B'])
        # !!! creating sub c matrix !!! (part 1/2: ppt, tmax, tmin, historical matrix, 2050-2099 ) note: we do not have monthly npp and vtype data, but I will extend them by fulfilling with duplicates monthly npp and vtype
        locals()[item+'_'+mdl+'_C_p1']=locals()[item+'_yr_mean'][0:(2015-1895),] # check results, C_sub matrices in here should be the same for each item 
        locals()['A_'+mdl]=np.append(locals()['A_'+mdl],locals()[item+'_'+mdl+'_A_p1'])
        locals()['B_'+mdl]=np.append(locals()['B_'+mdl],locals()[item+'_'+mdl+'_B_p1'])
        i=i+1
        # ppt, tmax, tmin loops done
    # construct sub matrices for npp and vtype
    ## create temporary variables to convoy historical data, npp and vtype from mc2 historical results, 10 models = the same; truncated  
    npp_a_p2=np.load(mc2_hist[0])[(1975-1895):(2015-1895),].mean(axis=0)
    locals()['A_'+mdl]=np.append(locals()['A_'+mdl],npp_a_p2) # np.append is used as above to avoid any unnecessary but potential dimension issues, reshape applied in the following
    vtype_a_p2=np.load(mc2_hist[1])[(1975-1895):(2015-1895),].mean(axis=0)
    locals()['A_'+mdl]=np.append(locals()['A_'+mdl],vtype_a_p2)
    locals()['A_'+mdl+'_agg']=locals()['A_'+mdl].reshape(5, len(dist_id_sel))
    ## load future data, model-specific, truncated
    npp_b_p2=np.load(locals()["future_n_p_tma_tmi_v_"+model_file][0])[2050-2099+1:,].mean(axis=0)
    locals()['B_'+mdl]=np.append(locals()['B_'+mdl],npp_b_p2)
    vtype_b_p2=np.load(locals()["future_n_p_tma_tmi_v_"+model_file][4])[2050-2099+1:,].mean(axis=0)
    locals()['B_'+mdl]=np.append(locals()['B_'+mdl],vtype_b_p2)
    locals()['B_'+mdl+'_agg']=locals()['B_'+mdl].reshape(5, len(dist_id_sel))
    m_control=m_control+1 # the initial value of m is outside of the loop, so 10 loops lead to m=10
    ## construct C matrices for npp and vtype
    locals()["npp_"+mdl+"_C_p2"]=np.load(mc2_hist[0])[0:(2015-1895),]
    locals()["vtype_"+mdl+"_C_p2"]=np.load(mc2_hist[1])[0:(2015-1895),]
    # data preparation done
    # PCA algothrimn begines
    dist=[]
    A_agg=locals()['A_'+mdl+'_agg']
    B_agg=locals()['B_'+mdl+'_agg']
    locals()[mdl+"_std_zero"]=[]
    for ref_dist in range(0,len(dist_id_sel)):
        ## Step 1: standardization of the climate data with reference period (1895-1994 on monthly basis) ICV at ICV proxy ref_dist
        B_j=B_agg[:,ref_dist]
        C_j=np.transpose(np.concatenate((locals()['ppt_'+mdl+'_C_p1'][:,ref_dist], locals()['tmax_'+mdl+'_C_p1'][:,ref_dist],locals()['tmin_'+mdl+'_C_p1'][:,ref_dist], locals()["npp_"+mdl+"_C_p2"][:,ref_dist], locals()["vtype_"+mdl+"_C_p2"][:,ref_dist] )).reshape(5,len(locals()['ppt_'+mdl+'_C_p1']))) #C_j should be T_mon X n(var), T X K in Mohany's paper 
        C_j_sd=C_j.std(axis=0)
        if C_j_sd[4]==0:
            locals()[mdl+"_std_zero"]=np.append(locals()[mdl+"_std_zero"], ref_dist)
            continue
        A_prime=A_agg/C_j_sd[:,None]
        B_j_prime=B_j/C_j_sd
        C_j_prime=C_j/C_j_sd
        ## Step 2: principal component analyses on the reference matrix C, and principal components extraction 
        C_j_prime_avg=np.mean(C_j_prime,axis=0)
        m, n = np.shape(C_j_prime)
        C_adj = []
        C_j_prime_p_avg = np.tile(C_j_prime_avg, (m, 1))
        C_adj = C_j_prime - C_j_prime_p_avg
        # calculate the covariate matrix
        covC = np.cov(C_adj.T)   
        # solve its eigenvalues and eigenvectors
        C_eigen_val, C_eigen_vec=  np.linalg.eig(covC)  
        # rank the eigenvalues: in here, I did not apply the truncation rule for the sake of limited variable availability
        index = np.argsort(-C_eigen_val)
        finalData = []
        # C matrix, corrected with PCA
        C_pca_vec=C_eigen_vec.T
        # A and B matrices, corrected with PCA
        X=A_prime.T.dot(C_pca_vec)
        Y_j=B_j_prime.T.dot(C_pca_vec)
        # project C to PCA
        Z_j=C_j_prime.dot(C_pca_vec)
        ## Step 3a: standarlization of anomalies 
        Z_j_sd=Z_j.std(axis=0)
        X_prime=X/Z_j_sd
        Y_j_prime=Y_j/Z_j_sd
        for int_loop in range(0,len(dist_id_sel)):
            distances[int_loop,ref_dist]=np.linalg.norm(X_prime[int_loop]-Y_j_prime)
            print(ref_dist,int_loop,m_control)
    min_dist_val=np.amin(distances,axis=0) 
    locals()[mdl+'min_dist_ind']=[]
    for min_ind in range(0,min_dist_val.shape[0]):    
          
        locals()[mdl+'min_dist_ind']=np.append(locals()[mdl+'min_dist_ind'],np.where(distances[:,min_ind]==np.amin(min_dist_val[min_ind])))
        print("Future", dist_names_valid[min_ind], "is analogous to current", dist_names_valid[int(locals()[mdl+'min_dist_ind'][min_ind])], file=f)        
f.close()
ending_clock = datetime.now()  
print(ending_clock-starting_clock)
# i7-6700k, 32GB, hyperthreading enabled, 0:11:20.197592
# R5 3550h, 16GB, hyperthreading enabled, 0:07:27.276289
  
    
  
    