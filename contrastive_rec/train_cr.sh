#!/usr/bin/env bash

for lr in 0.1 0.01 0.001 0.0001
 do
   for con_weight in 0.3
   do
     for reg_weight in 0.3
     do
       for temp_value in 1.5
       do
         nohup python cr_all.py --gpu 5 --lr $lr --con_weight $con_weight --reg_weight $reg_weight --temp_value $temp_value &
       done
     done
   done
 done