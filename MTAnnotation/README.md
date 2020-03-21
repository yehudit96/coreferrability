This folder contain all you need for the Mechanical Turk Annotation.
The annotation has two stages - 
1. Creating the task
2. Pre-annotation: Creating the HIT csv files, each file with 50 samples.
3. Annotation: Check after each HIT the cappa-cohen score and if any of the workers need to be removed. 
4. Post-annotation: Concatenating the result of the HITs to one scv and the computing the results of the annotaions and saving to a dictionary in a pickled file.s

The relevant scripts:
1. Creating the task:
    a. Uploat the HTML file - AMT template.html
    b. create_qualification_test.py - create the qualification test for the task, pay attention to create for the sandbox and for the production.
2. Pre-annotation
     samples_for_annotation_task.py --data_split (train\dev\test\eval) --annotated_dir ../data/tweets/ --out {}_samples
     create the samples (with some annotated samples for annotation checking) and saving to out dir.
3. Annotation
    check_batch.py --results_file {results file csv path}
    After each HIT, run this script to check how well the worker annotated, the workers with low score will be printed (check actually their results before removing them!)
4. Post-annotation
    a. concateneta_results.py --results_file_head {resulte files heads}
       concatenate all the files that starts with results_file_head to one csv file called results_file_head
    b. compute_annotation.py --results_file {results_file} --out {final results file}
       compute the result from the annotation file - results_file and save to out file.
       
In order to add the new annotated rules to the dictionary with all the annotation, look at classifier/datasets/Kian/update_labels.py
