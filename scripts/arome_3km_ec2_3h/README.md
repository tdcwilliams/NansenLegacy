# Run ECMWF forecast to spin up ensemble runs

## Setup experiment directory

```
OUTPUT_FORECAST_DATA_DIR=/cluster/work/users/timill/nextsimf_forecasts #as in $HOME/pynextsimf.src
fcname=arome_3km_ec2_3h.fram
FC_ROOT_DIR=$OUTPUT_FORECAST_DATA_DIR/$fcname
mkdir -p $FC_ROOT_DIR/20180209/restart
cp runtime/slurm.self_submitting.sh $FC_ROOT_DIR
cp $NEXTSIMDIR/model/bin/nextsim.exec $FC_ROOT_DIR
```

At time of writing CS2-SMOS initialisation was not working with intel-compiled
model. Therefore make a restart for `20180210` on another machine and copy the restart to
`$FC_ROOT_DIR/20180209/restart`.


## Start the forecast
Run 1-day forecasts with assimilation of Osisaf SSMI/AMSR2 concentrations
```
cd $FC_ROOT_DIR
sbatch slurm.self_submitting.sh
```
This effectively does
```
source $HOME/pynextsimf.src
exe=$FC_ROOT_DIR/nextsim.exec
rerun_forecast.py $fcname 20180210 20180331 1 -c -e $exe -ds -1
```
