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
yr_concurrent = yr0 = datetime.strptime("1895", '%Y').date()
yr1 = datetime.strptime("2099", "%Y").date()
yr_total=[yr_concurrent.strftime("%Y")]
while yr_concurrent < yr1:
    yr_concurrent =yr_concurrent + relativedelta(years=1)
    yr=datetime.strftime(yr_concurrent, "%Y")
    yr_total.append(yr)
    
dist_na=np.load("C:/Users/haoli/OneDrive/Work/Analogy/Python codes/sel_data/error_0.npy")
dist_id_all=list(range(0,503))
dist_id_sel=[e for e in dist_id_all if e not in dist_na]
dist_id_sel=list(dist_id_sel)
dist_names_valid=np.load("C:/Users/haoli/OneDrive/Work/Analogy/Python codes/analog/dist_n_a.npy")
dist_names_all=np.load("C:/Users/haoli/OneDrive/Work/Analogy/Python codes/analog/dist_n_all.npy")
dist_names_id=np.load('C:/Users/haoli/OneDrive/Work/Analogy/Python codes/analog/dist_id_valid.npy')
f_d=f_n=f_n_ext=[]
f = open('results', 'w')
f_f_d_na=open('future_dist_names', 'w')
f_f_d_id=open('future_dist_id', 'w')
f_c_d_na=open('current_dist_names', 'w')
f_c_d_id=open('current_dist_id', 'w')
f_mdl=open('in_models','w')
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
mdl_c=0  # set model control index
for mdl in model:
    i=0 # set item control index
    print(mdl)
    locals()['sp_'+mdl]=np.array([])
    # loop for ppt, tmax, tmin that has MACA historical inputs with 10 simulation models
    locals()['A_'+mdl]=[];locals()['B_'+mdl]=[] 
    for item in ["ppt", "tmax", "tmin"]:
        # print(item)
        # load historical data
        locals()[item+'_1895_1950']=np.load(prism_1_hist[i]) # load historical data (PRISM series)
        locals()[item+'_1951_2014']=np.load(prism_2_hist[i]) # load historical data (PRISM series)
        locals()[item+'_h']=np.concatenate((locals()[item+'_1895_1950'],locals()[item+'_1951_2014'])) # combine two-stage dataset into a historical data set, time range: 1895-2014
        # load future data, time range: 2006-2099
        locals()[item+"_f_a_"+mdl]=np.load(ppt_future[mdl_c])
        locals()[item+"_f_"+mdl]=np.delete(locals()[item+"_f_a_"+mdl],[it for it in range(0,(2015-2006)*12) ],0)
        # stack historical and future data
        locals()[item+"_all_"+mdl]=np.concatenate((locals()[item+'_h'], locals()[item+"_f_"+mdl]), axis=0)
        # calculate seasonal data, time ra
        locals()[item+'_mon_fmt_'+mdl]=locals()[item+"_all_"+mdl].reshape((int(locals()[item+"_all_"+mdl].shape[0]/12),12,len(dist_id_sel)))
        locals()[item+'_seasonal_all_'+mdl]=np.array_split(locals()[item+'_mon_fmt_'+mdl],4,axis=1)
        locals()[item+'_sp_'+mdl]=locals()[item+'_seasonal_all_'+mdl][0].mean(axis=1)
        locals()[item+'_sm_'+mdl]=locals()[item+'_seasonal_all_'+mdl][1].mean(axis=1)
        locals()[item+'_fl_'+mdl]=locals()[item+'_seasonal_all_'+mdl][2].mean(axis=1)
        locals()[item+'_wt_'+mdl]=locals()[item+'_seasonal_all_'+mdl][3].mean(axis=1)
        # construct A matrix (without npp): 1980-2014
        locals()[item+'_sp_'+mdl+"_A"]=locals()[item+'_sp_'+mdl][yr_total.index("1980"):yr_total.index("2015"),].mean(axis=0)
        locals()[item+'_sm_'+mdl+"_A"]=locals()[item+'_sm_'+mdl][yr_total.index("1980"):yr_total.index("2015"),].mean(axis=0)
        locals()[item+'_fl_'+mdl+"_A"]=locals()[item+'_fl_'+mdl][yr_total.index("1980"):yr_total.index("2015"),].mean(axis=0)
        locals()[item+'_wt_'+mdl+"_A"]=locals()[item+'_wt_'+mdl][yr_total.index("1980"):yr_total.index("2015"),].mean(axis=0)
        locals()[item+'_'+mdl+"_A_12vs"]=np.hstack((locals()[item+'_sp_'+mdl+"_A"],locals()[item+'_sm_'+mdl+"_A"],locals()[item+'_fl_'+mdl+"_A"],locals()[item+'_wt_'+mdl+"_A"]))
        # locals()['sp_'+mdl]=np.vstack([locals()['sp_'+mdl], locals()[item+'_sp_'+mdl]]) if locals()['sp_'+mdl].size else locals()[item+'_sp_'+mdl]
        # construct B matrix (without npp); stage 1: 2025-2050 
        locals()[item+'_sp_'+mdl+"_B1"]=locals()[item+'_sp_'+mdl][yr_total.index("2025"):yr_total.index("2051"),].mean(axis=0)
        locals()[item+'_sm_'+mdl+"_B1"]=locals()[item+'_sm_'+mdl][yr_total.index("2025"):yr_total.index("2051"),].mean(axis=0)
        locals()[item+'_fl_'+mdl+"_B1"]=locals()[item+'_fl_'+mdl][yr_total.index("2025"):yr_total.index("2051"),].mean(axis=0)
        locals()[item+'_wt_'+mdl+"_B1"]=locals()[item+'_wt_'+mdl][yr_total.index("2025"):yr_total.index("2051"),].mean(axis=0)
        locals()[item+'_'+mdl+"_B1_12vs"]=np.hstack((locals()[item+'_sp_'+mdl+"_B1"],locals()[item+'_sm_'+mdl+"_B1"],locals()[item+'_fl_'+mdl+"_B1"],locals()[item+'_wt_'+mdl+"_B1"]))
        # construct B matrix (without npp); stage 2: 2051-2075 
        locals()[item+'_sp_'+mdl+"_B2"]=locals()[item+'_sp_'+mdl][yr_total.index("2051"):yr_total.index("2076"),].mean(axis=0)
        locals()[item+'_sm_'+mdl+"_B2"]=locals()[item+'_sm_'+mdl][yr_total.index("2051"):yr_total.index("2076"),].mean(axis=0)
        locals()[item+'_fl_'+mdl+"_B2"]=locals()[item+'_fl_'+mdl][yr_total.index("2051"):yr_total.index("2076"),].mean(axis=0)
        locals()[item+'_wt_'+mdl+"_B2"]=locals()[item+'_wt_'+mdl][yr_total.index("2051"):yr_total.index("2076"),].mean(axis=0)
        locals()[item+'_'+mdl+"_B2_12vs"]=np.hstack((locals()[item+'_sp_'+mdl+"_B2"],locals()[item+'_sm_'+mdl+"_B2"],locals()[item+'_fl_'+mdl+"_B2"],locals()[item+'_wt_'+mdl+"_B2"]))
        # construct B matrix (without npp); stage 3: 2076-2099 
        locals()[item+'_sp_'+mdl+"_B3"]=locals()[item+'_sp_'+mdl][yr_total.index("2076"):yr_total.index("2099")+1,].mean(axis=0)
        locals()[item+'_sm_'+mdl+"_B3"]=locals()[item+'_sm_'+mdl][yr_total.index("2076"):yr_total.index("2099")+1,].mean(axis=0)
        locals()[item+'_fl_'+mdl+"_B3"]=locals()[item+'_fl_'+mdl][yr_total.index("2076"):yr_total.index("2099")+1,].mean(axis=0)
        locals()[item+'_wt_'+mdl+"_B3"]=locals()[item+'_wt_'+mdl][yr_total.index("2076"):yr_total.index("2099")+1,].mean(axis=0)                      
        locals()[item+'_'+mdl+"_B3_12vs"]=np.hstack((locals()[item+'_sp_'+mdl+"_B3"],locals()[item+'_sm_'+mdl+"_B3"],locals()[item+'_fl_'+mdl+"_B3"],locals()[item+'_wt_'+mdl+"_B3"]))
        # construct C matrix ICV (ICV, without npp): 1980-2014 
        locals()[item+'_sp_'+mdl+"_C"]=locals()[item+'_sp_'+mdl][yr_total.index("1980"):yr_total.index("2015"),]
        locals()[item+'_sm_'+mdl+"_C"]=locals()[item+'_sm_'+mdl][yr_total.index("1980"):yr_total.index("2015"),]
        locals()[item+'_fl_'+mdl+"_C"]=locals()[item+'_fl_'+mdl][yr_total.index("1980"):yr_total.index("2015"),]
        locals()[item+'_wt_'+mdl+"_C"]=locals()[item+'_wt_'+mdl][yr_total.index("1980"):yr_total.index("2015"),]
        # add up the control index
        i=i+1
        # item-specific processing done
        # exit the inner loop
    # construct 12-variable A matrix (partial)
    locals()[mdl+"_A_12vs"]=np.hstack((locals()['ppt_'+mdl+"_A_12vs"],locals()['tmax_'+mdl+"_A_12vs"],locals()['tmin_'+mdl+"_A_12vs"]))
    # construct 12-variable B matrices (partial)
    locals()[mdl+"_B1_12vs"]=np.hstack((locals()['ppt_'+mdl+"_B1_12vs"],locals()['tmax_'+mdl+"_B1_12vs"],locals()['tmin_'+mdl+"_B1_12vs"]))
    locals()[mdl+"_B2_12vs"]=np.hstack((locals()['ppt_'+mdl+"_B2_12vs"],locals()['tmax_'+mdl+"_B2_12vs"],locals()['tmin_'+mdl+"_B2_12vs"]))
    locals()[mdl+"_B3_12vs"]=np.hstack((locals()['ppt_'+mdl+"_B3_12vs"],locals()['tmax_'+mdl+"_B3_12vs"],locals()['tmin_'+mdl+"_B3_12vs"]))
    # construct 12-variable C matrices (partial)
    # process npp
    # npp historical: 1895-2014, universal
    npp_h=np.load(mc2_hist[0])
    # npp future: 2015-2099, model-specific
    npp_f=np.load(locals()["future_n_p_tma_tmi_v_"+mdl][0])
    locals()["npp_"+mdl]=np.vstack((npp_h,npp_f))
    # construct npp A matrix
    locals()['npp_'+mdl+"_A"]=locals()["npp_"+mdl][yr_total.index("1980"):yr_total.index("2015"),].mean(axis=0)
    # construct B matrix (without npp); stage 1: 2025-2050 
    locals()['npp_'+mdl+"_B1"]=locals()["npp_"+mdl][yr_total.index("2025"):yr_total.index("2051"),].mean(axis=0)
    # construct B matrix (without npp); stage 2: 2051-2075 
    locals()['npp_'+mdl+"_B2"]=locals()["npp_"+mdl][yr_total.index("2051"):yr_total.index("2076"),].mean(axis=0)  
    # construct B matrix (without npp); stage 3: 2076-2099 
    locals()['npp_'+mdl+"_B3"]=locals()["npp_"+mdl][yr_total.index("2076"):yr_total.index("2099")+1,].mean(axis=0)
    # construct npp C matrix ICV (ICV, without npp): 1980-2014 
    locals()['npp_'+mdl+"_C"]=locals()["npp_"+mdl][yr_total.index("1980"):yr_total.index("2015"),]
    # finalize A matrix (full)
    locals()[mdl+"_A"]=np.hstack((locals()['ppt_'+mdl+"_A_12vs"],locals()['tmax_'+mdl+"_A_12vs"],locals()['tmin_'+mdl+"_A_12vs"],locals()['npp_'+mdl+"_A"])).reshape(13, len(dist_id_sel))
    # finalize B matrices (full, 3 stages respectively)
    locals()[mdl+"_B1"]=np.hstack((locals()['ppt_'+mdl+"_B1_12vs"],locals()['tmax_'+mdl+"_B1_12vs"],locals()['tmin_'+mdl+"_B1_12vs"],locals()['npp_'+mdl+"_B1"])).reshape(13, len(dist_id_sel))
    locals()[mdl+"_B2"]=np.hstack((locals()['ppt_'+mdl+"_B2_12vs"],locals()['tmax_'+mdl+"_B2_12vs"],locals()['tmin_'+mdl+"_B2_12vs"],locals()['npp_'+mdl+"_B2"])).reshape(13, len(dist_id_sel))
    locals()[mdl+"_B3"]=np.hstack((locals()['ppt_'+mdl+"_B3_12vs"],locals()['tmax_'+mdl+"_B3_12vs"],locals()['tmin_'+mdl+"_B3_12vs"],locals()['npp_'+mdl+"_B3"])).reshape(13, len(dist_id_sel))
    # final preparation for PCA correction
    A_agg=locals()[mdl+"_A"]
    B1_agg=locals()[mdl+"_B1"]
    B2_agg=locals()[mdl+"_B2"]
    B3_agg=locals()[mdl+"_B3"]
    print(mdl_c)
    B_ctrl=0
    for B_agg in (locals()[mdl+"_B1"], locals()[mdl+"_B2"], locals()[mdl+"_B3"]):
        # print(B_agg)
        for ref_dist in range(0,len(dist_id_sel)):
            ## Step 1: standardization of the climate data with reference period (1895-1994 on monthly basis) ICV at ICV proxy ref_dist
            B_j=B_agg[:,ref_dist]
            C_j=np.transpose(np.concatenate((locals()['ppt_sp_'+mdl+'_C'][:,ref_dist], locals()['ppt_sm_'+mdl+'_C'][:,ref_dist],locals()['ppt_fl_'+mdl+'_C'][:,ref_dist], locals()['ppt_wt_'+mdl+'_C'][:,ref_dist], locals()['tmax_sp_'+mdl+'_C'][:,ref_dist], locals()['tmax_sm_'+mdl+'_C'][:,ref_dist],locals()['tmax_fl_'+mdl+'_C'][:,ref_dist], locals()['tmax_wt_'+mdl+'_C'][:,ref_dist],locals()['tmin_sp_'+mdl+'_C'][:,ref_dist], locals()['tmin_sm_'+mdl+'_C'][:,ref_dist],locals()['tmin_fl_'+mdl+'_C'][:,ref_dist], locals()['tmin_wt_'+mdl+'_C'][:,ref_dist],  locals()['npp_'+mdl+"_C"][:,ref_dist] )).reshape(13,len(locals()['ppt_sp_'+mdl+'_C']))) #C_j should be T_mon X n(var), T X K in Mohany's paper 
            C_j_sd=C_j.std(axis=0)
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
            index = np.argsort(-C_eigen_val) # equal to index = eigenValues.argsort()[::-1]   
            C_eigen_val=C_eigen_val[index]
            C_eigen_vec=C_eigen_vec[:,index]
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
                print(ref_dist,int_loop,B_ctrl, mdl_c)    
        min_dist_val=np.amin(distances,axis=0) 
        min_dist_ind=[]
        for min_ind in range(0,min_dist_val.shape[0]):    
            min_dist_ind=np.append(min_dist_ind,np.where(distances[:,min_ind]==np.amin(min_dist_val[min_ind])))
            #print("Future", dist_names_valid[min_ind], "is analogous to current", dist_names_valid[int(locals()[mdl+'min_dist_ind'][min_ind])], file=f)   
            print(dist_names_valid[min_ind], file=f_f_d_na)
            print(min_ind, file=f_f_d_id)
            print(dist_names_valid[int(min_dist_ind[min_ind])], file=f_c_d_na)
            print(min_dist_ind[min_ind], file=f_c_d_id)
            print(mdl,file=f_mdl)
        B_ctrl+=1
    mdl_c+=1
    
f.close()
f_f_d_na.close()
f_f_d_id.close()
f_c_d_na.close()
f_c_d_id.close()
f_mdl.close()
# ending_clock = datetime.now()  
# print(ending_clock-starting_clock)

    
    # locals()[mdl+"_B1_12vs"]=np.hstack((locals()[mdl+"_A_12vs"],locals()['tmax_'+mdl+"_B1_12vs"],locals()['tmin_'+mdl+"_B1_12vs"])).reshape(12, len(dist_id_sel))
    # locals()[mdl+"_B2_12vs"]=np.hstack((locals()['ppt_'+mdl+"_B2_12vs"],locals()['tmax_'+mdl+"_B2_12vs"],locals()['tmin_'+mdl+"_B2_12vs"])).reshape(12, len(dist_id_sel))
    # locals()[mdl+"_B3_12vs"]=np.hstack((locals()['ppt_'+mdl+"_B3_12vs"],locals()['tmax_'+mdl+"_B3_12vs"],locals()['tmin_'+mdl+"_B3_12vs"])).reshape(12, len(dist_id_sel))    
ending_clock = datetime.now()
print(ending_clock-starting_clock)  # 0:29:58.001244 - 0:30:18.599157 (i7-6700k hyp-dis); 0:56:49.791044 (i7-4810mq hyp_en); 0:32:52.920989 (R5-3550H)


    



















