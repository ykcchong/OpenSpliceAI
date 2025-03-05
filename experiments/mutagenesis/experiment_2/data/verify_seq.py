
file_path = "/ccb/cybertron/smao10/openspliceai/experiments/mutagenesis/experiment_2/data/donor_7.fa"
flanking = 10000

with open(file_path, 'r') as f:
    count_dict = {}
    for line in f.readlines():
        if line.startswith('>'):
            continue
        seq = str(line).strip()
        offset = (flanking // 2) + 199
        window = 2
        motif = seq[offset:offset+window]
        if motif in count_dict:
            count_dict[motif] += 1
        else:
            count_dict[motif] = 1

print(count_dict)