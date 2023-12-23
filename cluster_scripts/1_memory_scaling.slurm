#!/bin/bash

# load profile
source /home/gridsan/akaruvally/.bashrc

#conda init

hostname
# cd to the correct path
cd /home/gridsan/akaruvally/gsemm/gsemm_cpp/build/bin

######### VERIFY THIS BEFORE SCRIPT SUBMISSION

INSTANCE_ID=0
MAX_JOBS=8 # this should be set to the total number of jobs that gets created (in triples mode, it is p1*p2)

########################

((TRIAL_ID=0))
OUTPUT_PATH="journal/exp_gsemm_v2"

echo "EXPERIMENT: INSTANCE: ${INSTANCE_ID} TRIAL START: ${TRIAL_ID}"
echo "LLSUB_RANK: ${LLSUB_RANK}"

for N_NEURONS in $(seq 2 10 100);
do
  ((LOOP_ID=TRIAL_ID % MAX_JOBS))
  echo "${LOOP_ID} == ${LLSUB_RANK}"
  if [ $LOOP_ID -eq $LLSUB_RANK ]
  then
        EXPERIMENT_ID=$N_NEURONS
        echo "POLY_N: ${POLY_N} ALPHA_C: ${ALPHA_C}"
        echo "${EPOCHS} ${PER_SAMPLE_LEARNING_STEPS} ${ETA} ${L2_PENALTY}"

        ./1_0_expSM_testRun --diagnostic_timescale 1 --alpha_c 2.0 --T_d 0.05 --T_f 0.001 \
                            --max_seq_length 100000 --experiment_id $EXPERIMENT_ID --path $OUTPUT_PATH \
                            --n_neurons $N_NEURONS --no_phi
  fi
  ((TRIAL_ID=TRIAL_ID+1))
done
