#!/bin/bash

DIR="mag_vec_fem/data/20230522_charon"
DIR_TO_SAVE=${DIR}/plots
mkdir DIR_TO_SAVE
ETA="1e-1"
BETALIST="1e-1 2e-1 5e-2 1e-2"
#TARGET_LIST="0 "
#BETALIST="0 1e-2 2e-2 5e-2 1e-1 2e-1 5e-1 1 2"
TARGET_LIST="0 80 100"
for beta in $BETALIST
do
  NAME_U=u_Na400x15sig22v0L200eta${ETA}beta${beta}h1000
  NAME_W=w_Na400x15sig22v0L200eta${ETA}beta${beta}h1000
  LOAD_FILE_U=${DIR}/${NAME_U}.npz
  python3 mag_vec_fem/ungauged_landscape_plot.py eta=$ETA beta=$beta dir_to_save=$DIR_TO_SAVE load_file=$LOAD_FILE_U name_u=$NAME_U name_w=$NAME_W
  for target in $TARGET_LIST
  do
    NAME_FILE=Na400x15sig22v0eta${ETA}beta${beta}Symtarget${target}h1000Neig10.npz
    LOAD_FILE=${DIR}/${NAME_FILE}
    python3 mag_vec_fem/plotsfromdata.py eta=$ETA beta=$beta dir_to_save=$DIR_TO_SAVE load_file=$LOAD_FILE name_eig=$NAME_FILE
  done
done