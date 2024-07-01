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

** Do not run in predict-all or debug modes

## params
- flanking size: 80, 400, 2000, 10000
- without vs with annotation
- pytorch vs. keras spliceai model

16 runs