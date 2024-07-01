# benchmarking
1. CPU time, GPU time
2. I/O reads and writes
3. Memory usage over time
4. GPU vs. CPU utilization
5. Copy volume

# commands
scalene [-options] variant_test.py [-program_args]

**NOTE: our implementation of variant supports both keras and pytorch models by default, no need to create separate script for that. 

## task
variant calling on Mills+1000G gold standard vcf file

## params
- flanking size: 80, 400, 2000, 10000
- pytorch vs. keras spliceai model

8 runs