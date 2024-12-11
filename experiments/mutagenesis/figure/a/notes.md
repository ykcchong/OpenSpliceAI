# Data
Sample: 4 (U2SURP exon 9 +, GRCh37)
Location: chr3:142,740,137-142,740,263 (127 nt)
Acceptor Site: chr3:142,740,192 (pos 55 in sequence)
End of Exon: chr3:142,740,227 (pos 90 in sequence)

Sample: 5 (DST exon 2 -, GRCh38)
Location: chr6:56,735,192-56,735,344 (153 nt)
End of Exon: chr6:56,735,228 (pos 116 in sequence)
Acceptor Site: chr6:56,735,289 (pos 55 in sequence)

All samples have 10k flanking sequence to yield most accurate calculations. 

# Method
Mutate each base individually in the sequence to every other base. For each mutated sequence, input to the model and collect the acceptor site score. Then, calculate the difference between the mutated and reference sequence acceptor site scores. Calculate an importance score for the base position (as referenced in the SpliceAI paper) by computing the difference between the reference sequence score at the position and the average of all four base scores (including the reference base and the three point mutations) at that position. 

# Visualization
We visualize the acceptor site importance scores of each position in a DNA logo of the reference sequence around the acceptor site.

"Impact of in silico mutating each nucleotide around exon 9 in the U2SURP gene. The vertical size of each nucleotide shows the decrease in the predicted strength of the acceptor site (black arrow) when that nucleotide is mutated (Delta score)."
