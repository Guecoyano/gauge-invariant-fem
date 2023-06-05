#!/bin/bash

H_1=1000
ETA=1e-3
NEIG=10
GAUGE="Sym"
NAMEPOT="Na400x15sig22v0"
L=200

today=$( date +%Y%m%d )   # or: printf -v today '%(%Y%m%d)T' -1
number=0

fname="${today}"
DIR0="mag_vec_fem/data/${fname}"
DIR=$DIR0
while [ -e "$DIR" ]; do
    printf -v DIR '%s-%02d' "$DIR0" "$(( ++number ))"
done
DIR="mag_vec_fem/data/20230601"
#mkdir "$DIR"
DIR_TO_SAVE=${DIR}/plots
#mkdir "$DIR_TO_SAVE"
#cp script.sh ${DIR}/script

#BETA="0 1e-2 5e-2 1e-1 2e-1 5e-1 1 2"
BETA="0 5e-2"
#TARGET="0 80 100 150"
TARGET="0"

for beta in $BETA
do
  NAME_U="u_${NAMEPOT}L${L}eta${ETA}beta${beta}h${H_1}"
  NAME_W="w_${NAMEPOT}L${L}eta${ETA}beta${beta}h${H_1}"
  LOAD_FILE=${DIR}/${NAME_U}.npz
  #python3 mag_vec_fem/data_one_computation.py beta=$beta  eta=$ETA Neig=$NEIG dir_to_save=$DIR h=1/$H_1 gauge=$GAUGE u=True eig=False name_u=$NAME_U
  #python3 mag_vec_fem/ungauged_landscape_plot.py h=1/${H_1} eta=$ETA beta=$beta dir_to_save=$DIR_TO_SAVE load_file=$LOAD_FILE name_u=$NAME_U name_w=$NAME_W
  for target in $TARGET
  do
    NAME_EIG="${NAMEPOT}eta${ETA}beta${beta}${GAUGE}target${target}h${H_1}Neig${NEIG}"
    python3 mag_vec_fem/data_one_computation.py beta=$beta  eta=$ETA Neig=$NEIG dir_to_save=$DIR h=1/$H_1 gauge=$GAUGE name_eig="${NAME_EIG}" target_energy="${beta}+0.01*${target}*${ETA}"

  #  NAME_FILE=Na400x15sig22v0eta${ETA}beta${beta}Symtarget${target}h${H_1}Neig10.npz
  #  LOAD_FILE=${DIR}/${NAME_FILE}
  #  python3 mag_vec_fem/plotsfromdata.py h=1/${H_1} eta=$ETA beta=$beta dir_to_save=$DIR_TO_SAVE load_file=$LOAD_FILE name_eig=$NAME_FILE
  done
done
