# Experiment Outlines

## Scoring schemes
1. **Calculate positional weight matrix for all 4 bases** (simple difference in score)
   - Generates a DNA logo
2. **Calculate averaged score change** (ref - avg of 4 bases)
   - Generates a line graph

## Experiments
1. **Gene mutation at single position**
   - Take score change at every position, only 3 will have nonzero delta score
2. **Window around acceptor (or donor) site and mutate every base**
3. **Branch point insertion** (focus on acceptor site) -> Replace the sequence

---

## mutagenesis.py

### Workflow
- Read genomic sequences from file
- Create a dataframe with the length of the sequence and rows for each mutation (positional weight matrix)
- Mutate from left to right each base
  - Calculate score change for both donor and acceptor scores

### Variables
- **Input genomic sequence:** Fasta file with transcripts for each sequence you want to examine
- **Mutation position:** Location of mutation, or a sliding filter to mutate every position
- **Mutated bases:** A list of base(s) that you want to overwrite at location (if it matches existing pattern, will just report a delta 0)
- **Scoring positions:** Location(s) where you want to log the score change


### Testing
