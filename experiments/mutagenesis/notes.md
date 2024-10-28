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
<<<<<<< HEAD
   - Figure 1.D
   - chr3:142740137-142740263 (127 nt) acceptor site -> sample 4acc 
      - THIS IS GRCh37!!! exon 9 from U2SURP gene (location changed in GRCh38)
3. **Branch point insertion** (focus on acceptor site) -> Replace the sequence
   - Figure S2.C
   - window around 100nt
   - 14289 test set of splice acceptors

NEW 3. Replicating results from figure 1D (similar to Exp 2)
- branch point prediction is scrapped

---

# Final Mutagenesis Figure

A. DNA logo - The averaged results from all 100 samples (exp 2) -> comparison of keras vs pytorch to show similarity
B. DNA logo - reproduce 1.D (exp 2)
C. full gene before vs. after (maybe bam coverage) - Show capturing new cryptic splice site (exp 1)
D. bar plot - reproduce S2.C (exp 3)

=======
3. **Branch point insertion** (focus on acceptor site) -> Replace the sequence

---

>>>>>>> main
<!-- ## mutagenesis.py

### Workflow
- Read genomic sequences from file
- Create a dataframe with the length of the sequence and rows for each mutation (positional weight matrix)
- Mutate from left to right each base
  - Calculate score change for both donor and acceptor scores

### Variables
- **Input genomic sequence:** Fasta file with transcripts for each sequence you want to examine
- **Mutation position:** Location of mutation, or a sliding filter to mutate every position
- **Mutated bases:** A list of base(s) that you want to overwrite at location (if it matches existing pattern, will just report a delta 0)
- **Scoring positions:** Location(s) where you want to log the score change -->

---

## Experiment 1 
Mutate a donor/acceptor site and see how all nearby bases change score (notably, if donor/acceptor gain elsewhere, donor/acceptor loss at site)

Inputs: 
- fasta file (specified by params) 
   - site (donor or acceptor)
   - sample number
   *format requirements* 
      - must be a singular transcript
      - is a complete protein-coding gene containing splice site(s)
      - has a header line
- model (specified by params)
   - model type
   - flanking size
- experiment number (for output file writing)
- scoring position (the earliest base position where mutation should start)
- mutation length (the length of the mutation, default: 2)

Workflow: 
- 

Outputs:
- dna logo
   - the reference sequence with height proportional to average change in score when splice site mutated (over all 15 possible mutations)


---

## Experiment 2 
Mutate a 400bp window with the donor/acceptor site at the middle, and see how each position's mutations affects the overall change in score of the donor/acceptor site.

Inputs: 
- fasta file (specified by params) 
   - site (donor or acceptor)
   - sample number
   *format requirements* 
      - can have more than one transcript
      - input is 400bp with the splice site in the middle (pos 199 and 200)
      - has a header line
- model (specified by params)
   - model type
   - flanking size
- experiment number (for output file writing)
- scoring position (default: 198 donor, 201 acceptor)

Workflow: 
- iterates over every sequence
   - iterates over each base in the sequence
      - generates all 4 versions (3 mutated, 1 original) of the sequence
      - gets model's score of sequence and extracts donor and acceptor score at scoring position
      - updates cumulative scores at the position across all sequences
         - additionally stores a reference score in dedicated row (which is the same as the original base, but helps when calculating delta)
      - updates counts of scores at each position and mutated base
- calculates average scores by dividing cumulative scores with counts
- calculates score changes relative to reference
- depending on if sequence was donor or acceptor, generates and stores outputs

Outputs:
- average score change plot
   - the difference between average score of all 4 versions and reference sequence, relative to base position
- dna logo
   - change in score of donor/acceptor site, relative to base position and mutation
- score csv file
   - the raw score for acceptor and donor positions, and change relative to reference