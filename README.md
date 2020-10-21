# Neural_Execution_Engines
Code for NeurIPS paper: Neural Execution Engines: Learning to Execute Subroutines

## Software requirement:
This code was tested with Python 3.7.4, numpy 1.17.1 and tensorflow 2.0.0-rc0

## Usage:
### Baselines:
To see how different architectural changes affect the transformer's performance, you can run the following command in the baselines folder:

python Transformer_variants_sel_sort.py -V [variants_name] -E [test_data_type] -R [reload_directory]

--att_sup: optional, choose whether to use mask supervision

The only arguments required is -V [variants_name].
To see different options:
python Transformer_variants_sel_sort.py -h

### NEE:
Please run the following commands in the folder run_exp

To see different options:
python run_experiment.py -h

#### General use:
python run_experiment.py -T [task_name] -R [reload_file_path] -H [number of holdout]

task_name options:

"add": addition task
"addhold": addition task with holdout
"mul": multiplication task
"merge": merge sort task
"sel": selection sort task
"predist": train Dijkstra's shortest path with graph traces
"premst": train Prim's minimum spanning tree with graph traces
"dist": evaluate Dijkstra's shortest path
"mst": evaluate Prim's minimum spanning tree

To note: run "dist" or "mst" require 2 file paths


To run graph algorithms, we need to first train it using:

Dijkstra: python run_experiment.py -T predist
Prim: python run_experiment.py -T premst 

Then do the evaluation:

Using selection sort and test on various graphs:
Prim: python run_experiment.py -T mst -R Shortest_path_05_03_00_52 Shortest_path_05_03_00_52
Dijkstra: python run_experiment.py -T dist -R Shortest_path_05_03_00_52 Shortest_path_05_03_00_52

Using ER traces to train and test on various graphs:
Prim: python run_experiment.py -T mst -R premst_09_03_48_55 premst_09_03_48_55
Dijkstra: python run_experiment.py -T dist -R Shortest_path_05_03_00_52 predist_06_19_22_32



#### Embedding visualizations:

The embeddings (one for each bit) are stored in the folder of current execution. First transform the emb.npy into emb.mat and then change the directory path in the corresponding m file.


Since codes to run different task share similarities, I will mainly comment on run_sel_sort.py file for your reference.

