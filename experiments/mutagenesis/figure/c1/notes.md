# Data
MYBPC3 gene (>chr11:47358709-47370709(-) (hg19))-> a 2k sequence which contains the mutations, with 10k flanking. The mutation is applied at the midpoint (position 47364709, G>A)

# Result
original gene coords (hg19): chr11:47,352,957-47,374,253 (-)
mutation: 47352957 -> in intron 14

Keras: 
- Reference
    - Cryptic Site: 1.6261594e-05
    - Canonical Site: 0.9984438
- Mutated
    - Cryptic Site: 0.9423555
    - Canonical Site: 0.07852324
- Delta
    - Cryptic Site: 0.9423392415046692
    - Canonical Site: -0.9199205636978149

PyTorch: 
- Reference
    - Cryptic Site: 1.441814e-07
    - Canonical Site: 0.999131
- Mutated
    - Cryptic Site: 0.9847335
    - Canonical Site: 0.015660666
- Delta
    - Cryptic Site: 0.9847334027290344
    - Canonical Site: -0.9834703803062439

# Method
Mutate the target base and examine the difference in acceptor site scores across the region. Similar to SpliceAI figure 2A, examine the scores at the two acceptor site gain and loss events, and compare with SpliceAI's original score differences. 

# Visualization
We visualize the gain and loss scores tabularly, showcasing the raw acceptor score before and after the mutation for both SpliceAI and OpenSpliceAI.