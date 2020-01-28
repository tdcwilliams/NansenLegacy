# Preparation of forcing files
## 1. Process the raw AROME files from met.no
eg `/Data/sim/data/AROME_barents_ensemble/raw/aro_eps_2018033100.nc`
has times
```
time = "2018-03-31 03", "2018-03-31 06", "2018-03-31 09", "2018-03-31 12", 
    "2018-03-31 15", "2018-03-31 18", "2018-03-31 21", "2018-04-01", 
    "2018-04-01 03", "2018-04-01 06", "2018-04-01 09", "2018-04-01 12", 
    "2018-04-01 15", "2018-04-01 18", "2018-04-01 21", "2018-04-02"
```
Some variables are instantaneous and some are accumulated from the start of the forecast.
Run
```
./process_arome_ensemble.py 20180331
```
Now
eg `/Data/sim/data/AROME_barents_ensemble/processed/aro_eps_20180331.nc`
has times
```
 time = "2018-03-31", "2018-03-31 03", "2018-03-31 06", "2018-03-31 09", 
    "2018-03-31 12", "2018-03-31 15", "2018-03-31 18", "2018-03-31 21", 
    "2018-04-01", "2018-04-01 03", "2018-04-01 06", "2018-04-01 09", 
    "2018-04-01 12", "2018-04-01 15", "2018-04-01 18", "2018-04-01 21", 
    "2018-04-02"
```
We got the 1st record from the forecast from the previous day.
We also deacculated the accumulated variables so that the new value is only accumulated for 1 time interval.

## 2. Download 3-hourly resolution forecasts from ECMWF:
```
./ecmwf_forecast_download_201803_3h.py
```
```
/Data/sim/data/AROME_barents_ensemble/ECMWF_forecast_arctic/legacy.1.nc
/Data/sim/data/AROME_barents_ensemble/ECMWF_forecast_arctic/legacy.2.nc
/Data/sim/data/AROME_barents_ensemble/ECMWF_forecast_arctic/legacy.3.nc
```

## 3. Process the ECMWF forecast files to make daily files
```
./ecmwf_forecast_download_201803_3h.py 20180331 \
    /Data/sim/data/AROME_barents_ensemble/ECMWF_forecast_arctic/ \
    /Data/sim/data/AROME_barents_ensemble/ECMWF_forecast_arctic/3h
```
This deaccumulates some variables.

## 4. Blend the AROME and ECMWF forecasts to extend the AROME spatial domain
```
./blend_ecmwf_arome_ensemble_ec2_3h.py 20180331
```
This gives
```
/Data/sim/data/AROME_barents_ensemble/blended/ec2_arome_blended_ensemble_20180331.nc
```
which has times
```
time = "2018-03-31", "2018-03-31 03", "2018-03-31 06", "2018-03-31 09", 
    "2018-03-31 12", "2018-03-31 15", "2018-03-31 18", "2018-03-31 21", 
    "2018-04-01", "2018-04-01 03", "2018-04-01 06", "2018-04-01 09", 
    "2018-04-01 12", "2018-04-01 15", "2018-04-01 18", "2018-04-01 21", 
    "2018-04-02" ;
```

## 5A. For running forecasts make a copy of the last time record and set its time to 3h after the last one.
This stops neXtSIM crashing at the last time step.
```
./make_fake_record.py 20180331
```
This gives
```
/Data/sim/data/AROME_barents_ensemble/blended_with_fake_record/ec2_arome_blended_ensemble_20180331.nc
```
which has times
```
time = "2018-03-31", "2018-03-31 03", "2018-03-31 06", "2018-03-31 09", 
    "2018-03-31 12", "2018-03-31 15", "2018-03-31 18", "2018-03-31 21", 
    "2018-04-01", "2018-04-01 03", "2018-04-01 06", "2018-04-01 09", 
    "2018-04-01 12", "2018-04-01 15", "2018-04-01 18", "2018-04-01 21", 
    "2018-04-02", "2018-04-02 03" ;
```

## 5B. For running free runs make a fake file ```ec2_arome_blended_ensemble_20180401.nc``` to stop neXtSIM crashing at last time step.
```
./make_fake_files.py
```
