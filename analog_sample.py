import numpy as np
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
"""
Note 1: 
data range
prism results time dimension: 1895-01-16 00:00:00 ... 1950-12-16 00:00:00 upper, 
                              1951-01-16 00:00:00 ... 2014-12-16 00:00:00 lower.
rcp results time dimension:  2006-01-15 2006-02-15 ... 2099-12-15

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
# index array: years
yr=np.arange(1895,2100)
# Processing PPT, Tmax, and Tmin, and creating matrices A (historical), B (analog pool), and C (reference, pre-correction of PCA, time series).
A=[];B=[];C=[]
distances=np.zeros(shape=(len(dist_id_sel),len(dist_id_sel)))
#min_dist=np.zeros(shape=(len(dist_id_sel),len(dist_id_sel)))
for item in ["ppt", "tmax", "tmin"]:
    locals()[item+'_1895_1950']=np.load("C:/Users/haoli/OneDrive/Work/Analogy/Python codes/sel_data/"+item+"_1895_1950_sel.npy")
    locals()[item+'_1951_2014']=np.load("C:/Users/haoli/OneDrive/Work/Analogy/Python codes/sel_data/"+item+"_1951_2014_sel.npy")
    locals()[item+'_h']=np.concatenate((locals()[item+'_1895_1950'],locals()[item+'_1951_2014']))
    locals()[item+'_yr_mean_p1']=locals()[item+'_h'].reshape((int(locals()[item+'_h'].shape[0]/12),12,len(dist_id_sel))).mean(axis=(1))
    yr_p1=len(locals()[item+'_yr_mean_p1'])+1894
    locals()[item+'_rcp85']=np.load("C:/Users/haoli/OneDrive/Work/Analogy/Python codes/sel_data/"+item+"_rcp85_sel.npy")
    locals()[item+'_yr_mean_p2']=locals()[item+'_rcp85'].reshape((int(locals()[item+'_rcp85'].shape[0]/12),12,len(dist_id_sel))).mean(axis=(1))
    locals()[item+'_yr_mean_p2']=locals()[item+'_yr_mean_p2'][-(2099-yr_p1):,:]
    locals()[item+'_yr_mean']=np.concatenate((locals()[item+'_yr_mean_p1'], locals()[item+'_yr_mean_p2']), axis=0)
    # creating A (historical) matrix: 1975-2014
    locals()[item+'_A']=locals()[item+'_yr_mean'][(1975-1895):(2015-1895),].mean(axis=0)
    ## yr_A=yr[(1975-1895):(2015-1895)]
    A=np.append(A,locals()[item+'_A'])
    # creating B (analog) matrix: 2050-2099 
    locals()[item+'_B']=locals()[item+'_yr_mean'][-50:,].mean(axis=0)
    ## yr_B=yr[-50:,]
    B=np.append(B,locals()[item+'_B'])
    # creating C (reference) underlying matrix on monthly basis: 1895-1994
    locals()[item+'_C']=locals()[item+'_h'][t_total.index("1895-01"):t_total.index("1995-01"),]
A_agg=A.reshape(3, len(dist_id_sel))
B_agg=B.reshape(3, len(dist_id_sel))
dist=[]
# processing the reference (C) matrix
for ref_dist in range(0,len(dist_id_sel)):
    ## Step 1: standardization of the climate data with reference period (1895-1994 on monthly basis) ICV at ICV proxy ref_dist
    B_j=B_agg[:,ref_dist]
    C_j=np.transpose(np.concatenate((ppt_C[:,ref_dist], tmax_C[:,ref_dist],tmin_C[:,ref_dist])).reshape(3,len(ppt_C))) #C_j should be T_mon X n(var), T X K in Mohany's paper 
    C_j_sd=C_j.std(axis=0)
    A_prime=A_agg/C_j_sd[:,None]
    B_j_prime=B_j/C_j_sd
    C_j_prime=C_j/C_j_sd
    ## Step 2: principal component analyses on the reference matrix C, and principal components extraction 
    C_j_prime_avg=np.mean(C_j_prime,axis=0)
    m, n=np.shape(C_j_prime)
    C_adj = []
    C_j_prime_p_avg=np.tile(C_j_prime_avg, (m, 1))
    C_adj=C_j_prime-C_j_prime_p_avg
    # calculate the covariate matrix
    covC=np.cov(C_adj.T)   
    # solve its eigenvalues and eigenvectors
    C_eigen_val, C_eigen_vec=np.linalg.eig(covC)  
    # rank the eigenvalues: in here, I did not apply the truncation rule for the sake of limited variable availability
    index=np.argsort(-C_eigen_val)
    finalData=[]
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
    ## Step 3b: find the Mahalanobis nearest neighbors, A.K.A. Euclidean nearest neighbors with a PCA-standarized correction matrix
    for int_loop in range(0,len(dist_id_sel)):
        distances[int_loop,ref_dist]=np.linalg.norm(X_prime[int_loop]-Y_j_prime)
        print(ref_dist,int_loop)
min_dist_val=np.amin(distances,axis=0) 
# load rangerlands districts names and indices
min_dist_ind=[]
dist_names_valid=np.load("C:/Users/haoli/OneDrive/Work/Analogy/Python codes/analog/dist_n_a.npy")
dist_names_all=np.load("C:/Users/haoli/OneDrive/Work/Analogy/Python codes/analog/dist_n_all.npy")
dist_names_id=np.load('C:/Users/haoli/OneDrive/Work/Analogy/Python codes/analog/dist_id_valid.npy')
# find smallest Mahalanobis distances, index them, and print the corresponding districts names.
for min_ind in range(0,min_dist_val.shape[0]):    
    min_dist_ind=np.append(min_dist_ind,np.where(distances[:,min_ind]==np.amin(min_dist_val[min_ind])))
    print("Future", dist_names_valid[min_ind], "is analogous to current",   dist_names_valid[int(min_dist_ind[min_ind])] )
ending_clock = datetime.now()  
print(ending_clock-starting_clock)
#starting datetime.datetime(2020, 6, 4, 18, 31, 24, 851147) i7-4810mq, 16GB
#ending datetime.datetime(2020, 6, 4, 18, 30, 18, 658403) i7-4810mq, 16GB, 66 seconds
#starting datetime.datetime(2020, 6, 4, 21, 47, 24, 848188) i7-6700k, 32GB
#ending datetime.datetime(2020, 6, 4, 21, 48, 4, 888163) i7-6700k, 32GB, 40 seconds
# R5-3550H, 16GB, 32 seconds
# i3-6100u, 8GB, 91 seconds
