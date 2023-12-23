#!/bin/bash

# load profile
source /home/gridsan/akaruvally/.bashrc

#conda init

hostname

######### VERIFY THIS BEFORE SCRIPT SUBMISSION

# cd to the correct path
cd /home/gridsan/akaruvally/gsemm/gsemm_cpp/build/bin

INSTANCE_ID=0

MAX_JOBS=384 # this should be set to the total number of jobs that gets created (in triples mode, it is p1*p2)

((TRIAL_ID=INSTANCE_ID*(100*49)+1))
((SEED_START=INSTANCE_ID*100))
((SEED_END=(INSTANCE_ID+1)*100))
OUTPUT_PATH="journal/3_0_1_exp_gsemm_LT"

########################

echo "EXPERIMENT: INSTANCE: ${INSTANCE_ID} TRIAL START: ${TRIAL_ID} SEEDS: ${SEED_START} - ${SEED_END}"
echo "LLSUB_RANK: ${LLSUB_RANK}"

while true
do
  SEED=$RANDOM
  ALPHA_C=$(bc <<< "scale=2; ($RANDOM*3.0/32767)+0.01")
  T_f=$(bc <<< "scale=3; ($RANDOM*0.01/32767)+0.001")
  Tl_xi=$(bc <<< "scale=0; ($RANDOM*100.0/32767)+1.0")
  Tl_phi=$(bc <<< "scale=0; ($RANDOM*100.0/32767)+1.0")
  T_d=$(bc <<< "scale=2; ($T_f*50)")
  STEPS_PER_MEMORY=$(bc <<< "scale=0; ($RANDOM*1000.0/32767)+1000")
  STEPS_PER_MEMORY=$(printf '%.0f\n' $STEPS_PER_MEMORY)

  echo "SEED: ${SEED} ALPHA_C: ${ALPHA_C} STEPS_PER_MEMORY: ${STEPS_PER_MEMORY}"
  echo "T_f: $T_f T_d: $T_d Tl_xi: $Tl_xi Tl_phi: $Tl_phi"

  ((LOOP_ID=TRIAL_ID % MAX_JOBS))
  echo "${LOOP_ID} == ${LLSUB_RANK}"

  if [ $LOOP_ID -eq $LLSUB_RANK ]
  then
        EXPERIMENT_ID=$TRIAL_ID

        ./3_0_1_expSM_LT_learning --diagnostic_timescale 50 --alpha_c $ALPHA_C --T_d $T_d --T_f $T_f \
                                    --Tl_xi $Tl_xi --Tl_phi $Tl_phi --steps_per_memory $STEPS_PER_MEMORY \
                                    --max_seq_length 10 --experiment_id $EXPERIMENT_ID --path $OUTPUT_PATH \
                                    --n_neurons 100 --epochs 20 --verbose
  fi
  ((TRIAL_ID=TRIAL_ID+1))
done
