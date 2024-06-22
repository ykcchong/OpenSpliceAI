from pyfaidx import Fasta
from Bio import SeqIO
import time

fasta_file = '/Users/alanmao/Desktop/Research/spliceAI-toolkit/train_data/human/hg19.fa'

print("--- SeqIO: reading fasta to dict ---")
start_time = time.time()
seq_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
print("--- %s seconds ---" % (time.time() - start_time))

print("--- pyfaidx: reading fasta to dict ---")
start_time = time.time()
genes = Fasta(fasta_file)
print("--- %s seconds ---" % (time.time() - start_time))