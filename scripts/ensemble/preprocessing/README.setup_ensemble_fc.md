# Setup ensemble forecast

## Spinup
Run with ECMWF ensemble forcing for 1 month (from 20180210)
in `/cluster/work/users/timill/nextsimf_forecasts/arome_3km_ec2_3h.fram`
to get restart for `20180309`. See
[this read-me
file](https://github.com/tdcwilliams/NansenLegacy/blob/master/scripts/arome_3km_ec2_3h/README.md)
for instructions on how to do this

## Setup
Create directories with restart and create config files from a template
```
EC2_3H_DIR=/cluster/work/users/timill/nextsimf_forecasts/arome_3km_ec2_3h.fram
RESTART_FORECAST_DATE_DIR=$EC2_3H_DIR/20180308
CFG_TEMPLATE=../config_files/arome_3km_ec2_arome_ensemble.fram.cfg
./setup_ensemble_fc.sh $CFG_TEMPLATE $RESTART_FORECAST_DATE_DIR
```
