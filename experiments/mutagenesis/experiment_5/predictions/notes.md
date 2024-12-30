# Data
TP53 gene, full length (chr17:7,570,739-7,591,808(-) (hg19)) -> 21069 bp sequence + 10k flanking context. 

# Method
First, we search for cryptic splicing events by using an initial single nucleotide mutation scan, calculating the importance score for each position in the gene. This score for a target is taken from summing the total number of positions whose score (acceptor or donor) change by >0.5 after a mutation at the target position, for all three possible mutations. By calculating the importance score for the TP53 gene and removing those at intron-exon boundaries, we locate deep intronic variants which putatively cause multiple high-impact splice site change events.  
Second, we mutate the target base and examine the difference in acceptor site scores across the region. Similar to SpliceAI figure 2A, examine the scores at the two acceptor site gain and loss events, and compare with SpliceAI's original score differences. We visualize this in an exon plot which shows donor and acceptor site score differences for both OpenSpliceAI and SpliceAI, and how the mutation will affect the exon plot. s

## NOTE: this experiment was tabled in favor of selecting well-documented examples from literature 