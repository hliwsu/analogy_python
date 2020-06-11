import xarray as xr 
import numpy as np
import regionmask
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4
import descartes
import os
import os.path
# Read rangelands boundary files (shapefile)
# my copy ~/haoli.li
# rb = gpd.read_file('/home/haoli.li/USDR/S_USA.RangerDistrict.shp')
# public copy ~/data/rajagopalan/MC2/USRD
rb = gpd.read_file("/data/rajagopalan/MC2/USRD/S_USA.RangerDistrict.shp")
rb.head()
# Obtain the number of rangeland districts 
rb_len=len(rb)
# Check if the number is correct
rb_len
f_d=[]
# not-available data array
n_v_r=[]
# obtain path and file name information of all *.nc files in '~/ConUS/Climate'
for dirpath, dirnames, filenames in os.walk("/data/rajagopalan/MC2/Harddrive_MC2/ConUS/Climate"):
    for filename in [f for f in filenames if f.endswith("ppt.nc")]:
        f_d.append(os.path.join(dirpath, filename))
all_f=len(f_d)
# loop over all of these *.nc files
for file in range(0,all_f):
    ncf = xr.open_dataset(f_d[file])
    print(ncf)
    yr_len=len(ncf.time)
    lat_n=len(ncf.lat)
    lon_n=len(ncf.lon)
    rb_mask_poly = regionmask.Regions_cls(name = 'Rangelands Distrcits', numbers = list(range(0,rb_len)), names = list(rb.DISTRICTNA), abbrevs = list(rb.DISTRICTOR), outlines = list(rb.geometry.values[i] for i in range(0,rb_len)))
    mask= rb_mask_poly.mask(ncf.isel(lat = slice(0,lat_n), lon = slice(0,lon_n)), lat_name='lat', lon_name='lon')
    # public values
    lat = mask.lat.values
    lon = mask.lon.values
    v_m=[]
    # a tentitative run to find out the deadspots 
    for r_d in range(0,rb_len):
        print("Rangeland district ID", r_d)
        print("Rangeland district name", rb.DISTRICTOR[r_d])
        try:
            sel_mask = mask.where(mask == r_d).values
            id_lon = lon[np.where(~np.all(np.isnan(sel_mask), axis=0))]
            id_lat = lat[np.where(~np.all(np.isnan(sel_mask), axis=1))]
            out_sel = ncf.sel(lat = slice(id_lat[0], id_lat[-1]), lon = slice(id_lon[0], id_lon[-1])).compute().where(mask == r_d)
            a=out_sel.ppt.values
            v_m.append(np.nanmean(a))
            print("Average climatic value is", np.nanmean(a))
        except: 
            n_v_r.append(r_d)
            v_m.append(np.nanmean(a))
            print("Unable to retrieve values on", rb.DISTRICTNA[r_d] , "marked on the error log")
        pass
    [rb.DISTRICTNA[i] for i in n_v_r]
    # formal data extract
    rb_a=list(range(0,rb_len))
    rb_a=[x for x in rb_a if x not in n_v_r]
    rb_a_len=len(rb_a)
    cl_data=np.empty((20,yr_len,rb_a_len))
    # loop years
    for yr in range(0,yr_len): 
        ncf_t=ncf.isel(time=slice(yr,yr+1))
        i=0
        # loop districts
        for r_d in rb_a:
            print("Rangeland district ID", r_d)
            print("Rangeland district name", rb.DISTRICTOR[r_d])
            sel_mask = mask.where(mask == r_d).values
            id_lon = lon[np.where(~np.all(np.isnan(sel_mask), axis=0))]
            id_lat = lat[np.where(~np.all(np.isnan(sel_mask), axis=1))]
            out_sel = ncf_t.sel(lat = slice(id_lat[0], id_lat[-1]), lon = slice(id_lon[0], id_lon[-1])).compute().where(mask == r_d)
            a=out_sel.ppt.values
            cl_data[0,yr,i]=np.nanmean(a)
            i=1+1
            print("Average climatic value is", np.nanmean(a), "in the year of", yr)
    np.save(str(file)+'.npy',cl_data)
