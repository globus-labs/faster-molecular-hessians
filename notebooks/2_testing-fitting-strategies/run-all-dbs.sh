#! /bin/bash

notebook=$1
dbs=$(find ../1_explore-sampling-methods/data/ -name "caffine_pm7_None*.db")
for db in $dbs; do
  echo $db
  papermill -p db_path $db $notebook -
done
