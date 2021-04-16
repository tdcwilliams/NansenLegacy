#! /bin/bash -x

if [ $# -lt 2 ]
then
    echo "Usage"
    echo "eval_freerun.sh INPUT_DIR MOORINGS_MASK [DO_OSISAF] [DO_DRIFT] [DO_SMOS] [DO_CS2SMOS]"
    exit 0
fi
ME=`readlink -f $0`
HERE=`dirname $ME`

do_copy=0 # copy moorings if running
FCDIR=$1
MOORINGS_MASK=${2-"Moorings_%Ym%m.nc"}
DO_OSISAF=${3-"1"}
DO_DRIFT=${4-"1"}
DO_SMOS=${5-"1"}
DO_CS2SMOS=${6-"0"}
numproc_sem=12

fcdir=$FCDIR/outputs
[[ ! -d $fcdir ]] && { echo $fcdir not present; exit 1; }

function run
{
    # if not on fram, launch normally
    nh=${#HOSTNAME}
    [[ "${HOSTNAME:$((nh-14)):14}" != "fram.sigma2.no" ]] && { sem -j $numproc_sem $1 & return; }
    cmd="singularity exec --cleanenv $PYNEXTSIM_SIF $1"
    $cmd; return #just run on login node

    # on fram, launch with sbatch
    mkdir -p logs
    args=($1)
    sbatch_opts=()
    sbatch_opts+=("--job-name=AN${args[0]}")
    sbatch_opts+=("--time=00:30:00")
    sbatch ${sbatch_opts[@]} --export=COMMAND="$cmd" $HERE/slurm_batch_job.sh
}

#copy Moorings to avoid causing a run to crash
fcdir=$FCDIR/outputs

# Inputs that are independant of the source
inputs="$fcdir -np -g arome_3km.msh"

if [ $DO_CS2SMOS -eq 1 ]
then
   smoothing="-sig 1"
   CMD="evaluate_forecast.py $inputs $smoothing -s Cs2SmosThick -mm $MOORINGS_MASK"
   odir="$FCDIR/eval-cs2smos"
   run "$CMD -o $odir"
fi

if [ $DO_OSISAF -eq 1 ]
then
   smoothing="-sig 2"
   CMD="evaluate_forecast.py $inputs $smoothing -s OsisafConc -mm $MOORINGS_MASK"
   odir="$FCDIR/eval-osisaf-conc"
   run "$CMD -o $odir"

   # extent
   CMD="evaluate_forecast.py $inputs $smoothing -s OsisafConc -ee -mm $MOORINGS_MASK"
   odir=$FCDIR/eval-osisaf-extent
   run "$CMD -o $odir"
fi

if [ $DO_SMOS -eq 1 ]
then
   smoothing="-sig 2"
   CMD="evaluate_forecast.py $inputs -nb 1 $smoothing -s SmosThick -mm $MOORINGS_MASK"
   odir="$FCDIR/eval-smos"
   run "$CMD -o $odir"
fi

if [ $DO_DRIFT -eq 1 ]
then
   inputs_drift="$fcdir -mu 2.5"
   #inputs_drift+=" -f"
   #inputs_drift="$FCDIR -np -g medium_arctic_10km.msh"
   CMD="evaluate_drift_forecast.py $inputs_drift -mm $MOORINGS_MASK"
   odir="$FCDIR/eval-osisaf-drift"
   run "$CMD -o $odir"

   inputs_drift="$fcdir -mu 20"
   #inputs_drift+=" -f"
   #inputs_drift="$FCDIR -np -g medium_arctic_10km.msh"
   CMD="evaluate_drift_forecast.py $inputs_drift -mm $MOORINGS_MASK"
   odir="$FCDIR/eval-osisaf-drift-mu10kpd"
   run "$CMD -o $odir"
fi
