# Data
OPA1 splice variant (chr3:193362516A>G) of MEP344
chr3:193362516A>G	c.1608+622A>G	heterozygous	splicing (in hg19 coords)
https://pmc.ncbi.nlm.nih.gov/articles/PMC7960924/
- located in intron 16 on forward strand of chr3 (AGTGAGGTAG[A]AACAAATTT)
- selected because it is in the testing data, and was shown to have the highest abberrant-only splicing in in vitro assays

CONVERTED TO GRCh38 coords:
chr3:193,644,727A>G 
- (AGTGAGGTAG[A]AACAAATTT) same neighborhood
- intron 16

# Result
Mutation: chr3:193,644,727A>G (2500)
Donor Gain: chr3:193,644,722 (2495)
Acceptor Gain: chr3:193,644,669 (2442)

Keras: 
- Reference
    - Donor: 0.24689993
    - Acceptor: 0.09340113
- Mutated
    - Donor: 0.98174703
    - Acceptor: 0.9696549
- Delta
    - Donor: 0.7348470687866211
    - Acceptor: 0.8762537837028503

PyTorch: 
- Reference
    - Donor: 0.34271866
    - Acceptor: 0.098811634
- Mutated
    - Donor: 0.98482853
    - Acceptor: 0.9887656
- Delta
    - Donor: 0.6421098709106445
    - Acceptor: 0.8899539709091187



