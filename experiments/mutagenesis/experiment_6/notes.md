# Data
MTRR gene - c.903+469T>C
https://pmc.ncbi.nlm.nih.gov/articles/PMC3429857/
coordinates: chr5:7,876,246-7,891,246

## Notes on HGVS nomenclature and calculating genomic coordinates
https://hgvs-nomenclature.org/stable/background/simple/
NM_002454.2:c.903+469T>C
- (reference):(cDNA).(1-based position)+(downstream intronic position)(orig base)>(new base)
- it is very important to ensure you have the right reference and for which genome it corresponds to (hg38) -> can look this up in NCBI
- the cDNA position is based on the *SPLICED OUT mRNA TRANSCRIPT*, so it is in terms of the aligned length of the transcript to the genome, not the gene length
    - this means that it will ignore introns as part of the length, hence the second part of the (+/-) index which specifically indicates to count into the intronic region
    - if it is the case that there is a second part, then the first part of the index should end at the end of an exon
- the 1-based indexing further starts with +1 on the *ATG* initiation codon, which may not correspond to the beginning of the transcript if there is a 5' UTR
Calculating the genomic coordinate:
- the above point mutation occurs at position (137 start codon + 903 cDNA + 469 into the intron)
- 137 + 903 = 1040 which is the end of the exon in the transcript, yielding (ggacatttca) as the last 10 bases -> this matches the end of exon 6 in GRCh38, at position 7883277
- then going 7883277 + 469 bases downstream from this exon 6 into intron 6, we get position 7883746 -> (AATGGC[T]GGAGGA), which is the T we expect, that will be mutated into C. 
- in order to get the full 10k window and all the relevant information for the neighboring splice sites, we extract a 5k sequence including exons 6-7 and the full intron 6, with 10k flanking
    - mutation position: 7,883,746
    - exon 6 - 7: 7,883,155-7,885,854
    - SL (centered on mutation): 7,881,246-7,886,246
    - full sequence: chr5:7,876,246-7,891,246