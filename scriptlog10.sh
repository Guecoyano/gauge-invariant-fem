#!/bin/bash

H_1=1000
ETA=2e-1
NEIG=10
GAUGE="Sym"
NAMEPOT="Na400x15sig22v0"
L=200
DIR="mag_vec_fem/data/20230601-01"
DIR_TO_SAVE=${DIR}/plots

#BETA="0 1e-2 5e-2 1e-1 2e-1 5e-1 1 2"
BETA="2e-1"
#TARGET="0 80 100 150"
TARGET="0 80 100"

for beta in $BETA
do
  NAME_U="u_${NAMEPOT}L${L}eta${ETA}beta${beta}h${H_1}"
  #python3 mag_vec_fem/data_one_computation.py beta=$beta  eta=$ETA Neig=$NEIG dir_to_save=$DIR h=1/$H_1 gauge=$GAUGE u=True eig=False name_u=$NAME_U
  for target in $TARGET
  do
    NAME_EIG="${NAMEPOT}eta${ETA}beta${beta}${GAUGE}target${target}h${H_1}Neig${NEIG}"
    #python3 mag_vec_fem/data_one_computation.py beta=$beta  eta=$ETA Neig=$NEIG dir_to_save=$DIR h=1/$H_1 gauge=$GAUGE name_eig="${NAME_EIG}" target_energy="${beta}+0.01*${target}*${ETA}"

    NAME_FILE=${NAME_EIG}.npz
    LOAD_FILE=${DIR}/${NAME_FILE}
    python3 mag_vec_fem/plotsfromdata.py h=1/${H_1} eta=$ETA beta=$beta dir_to_save=$DIR_TO_SAVE load_file=$LOAD_FILE name_eig=$NAME_FILE modulus=False log10=True
  done
done
