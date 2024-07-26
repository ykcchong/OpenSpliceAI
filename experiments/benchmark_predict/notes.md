# benchmarking
1. CPU time, GPU time
2. I/O reads and writes
3. Memory usage over time
4. GPU vs. CPU utilization
5. Copy volume

## tools
- pyperf
- timeit
- scalene **
- mprof
- memory_profiler

# commands
scalene [-options] predict_test.py [-program_args]

scalene [-options] spliceai_default_pred.py [-program_args]

## task
predict on grch38 chr1

## params
- flanking size: 80, 400, 2000, 10000
- without vs with annotation
- pytorch vs. keras spliceai model

16 runs

# special notes
- Do not run in predict-all or debug modes
- use the Live Preview (Microsoft) extension to preview the html outputs natively
- tested the cli version, does not have as much information, but can redirect all outputs to files

## extracting information from profile.json scalene output
- elapsed_time_sec
- locate the `predict_and_write` line (files -> "/ccb/cybertron/smao10/openspliceai/experiments/benchmark_predict/predict_test.py" -> functions -> line = "predict_and_write")
    "n_avg_mb": 10.307589530944824,
    "n_copy_mb_s": 0.0,
    "n_core_utilization": 0.030044843064412876,
    "n_cpu_percent_c": 1.1387174628407477,
    "n_cpu_percent_python": 0.150917624305348,
    "n_gpu_avg_memory_mb": 308.3333333333333,
    "n_gpu_peak_memory_mb": 342.0,
    "n_gpu_percent": 0,
    "n_growth_mb": 10.307589530944824,
    "n_malloc_mb": 10.307589530944824,
    "n_mallocs": 1,
    "n_peak_mb": 10.307589530944824,
    "n_python_fraction": 0.19941300000000003,
    "n_sys_percent": 2.970015096329677,
    "n_usage_fraction": 0.03215881989925033
- growth_rate
- max_footprint_mb
- samples -> plot memory over time

# debug
python predict_test.py -m ../../models/spliceai-mane/10000nt/model_10000nt_rs42.pt -o results_test_10k -f 10000 -i ./data/chr1.fa -a ./data/chr1_subset100.gff -t 0.9 -D > ./results_test_10k/output.log 2> ./results_test_10k/error.log

scalene predict_test.py -m ../../models/spliceai-mane/10000nt/model_10000nt_rs42.pt -o results_scalene_10k -f 10000 -i ./data/chr1.fa -a ./data/chr1_subset100.gff -t 0.9 -D > ./results_scalene_10k/output.log 2> ./results_scalene_10k/error.log

/usr/bin/time -v python  predict_test.py -m ../../models/spliceai-mane/10000nt/model_10000nt_rs42.pt -o results_scalene_10k -f 10000 -i ./data/chr1.fa -a ./data/chr1_subset100.gff -t 0.9 -D -p  > ./results_scalene_10k/output.log 2> ./results_scalene_10k/error.log

/usr/bin/time -v python spliceai_default_test.py -o results_time_10k -f 10000 -i ./data/chr1.fa -a ./data/chr1_subset100.gff -t 0.9  > ./results_time_10k/output.log 2> ./results_time_10k/error.log