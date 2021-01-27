#!/bin/bash

# Reset repository : remove all the log files, plots and benchmark results

while true; do
    read -p "!!! WARNING !!! This will remove all the log files, plots and benchmark results. This cannot be undone. Do you want to continue? [yN] " yn
    yn=${yn:-N}  # default answer is No
    case $yn in
        [Yy]* ) 
            echo "Deleting log files";
            rm -rf logs/
            mkdir logs
            rm -rf submission_logs/
            mkdir submission_logs
            echo "Deleting benchmark plots";
            rm -rf benchmark/plots/
            mkdir benchmark/plots/
            echo "Resetting benchmark results";
            rm -rf benchmark/benchmark_results.csv
            cp benchmark/.reset_backup/benchmark_results_blank.csv benchmark/
            mv benchmark/benchmark_results_blank.csv benchmark/benchmark_results.csv
            break;;
        [Nn]* ) 
            echo "Aborting..."; 
            exit;;
        * ) echo "Please answer y (yes) or n (no).";;
    esac
done

