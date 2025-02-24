# Description: Visualize the results of the mutagenesis experiment for the splice site prediction model.
import pandas as pd
import numpy as np
from spliceaitoolkit.constants import *
from vis import mutate_base, calculate_average_score_change, generate_dna_logo, plot_average_score_change
import itertools

##############################################
## MUTAGENESIS EXPERIMENT
##############################################

# Main function for mutagenesis experiment
def visualize(df, site, output_base):
    
    # Load the results DataFrame
    df.columns = ['identifier', 'position', 'ref_base', 'base', 'score1', 'score2']
    df['position'] = df['position'].astype(int)
    
    if site == 'acceptor':
        df['score'] = df['score1']
    else:
        df['score'] = df['score2']
    
    # Extract reference scores
    ref_df = df[df['base'] == 'ref'][['identifier', 'position', 'ref_base', 'score']]
    ref_df = ref_df.rename(columns={'score': 'ref_score'})
    
    # Merge the reference scores back into the main DataFrame
    df = df.merge(ref_df[['identifier', 'position', 'ref_score']], on=['identifier', 'position'], how='left')

    # Calculate the score change from reference
    df['score_change'] = df['ref_score'] - df['score']

    # Exclude rows where base is 'ref' as we only need the changes for other bases
    df = df[df['base'] != 'ref']

    # Group by position and base, then calculate the average score change across all sequences
    result_df = df.groupby(['position', 'base'])['score_change'].mean().reset_index()

    # Optional: Pivot the DataFrame for better readability
    pivot_df = result_df.pivot(index='position', columns='base', values='score_change').reset_index()
    
    print(pivot_df)

    # Generate DNA logo
    generate_dna_logo(pivot_df, site, f'{output_base}_dna_logo.png')
    print('saved to', f'{output_base}_dna_logo.png')

def mutagenesis():
        
    sites = ['donor', 'acceptor']
    flanking_sizes = [10000]

    base_dir = f'/ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/b/results/keras_job'
    
    for flanking_size, site in itertools.product(flanking_sizes, sites):
        
        # Aggregate all the results
        dataframes = []
        for i in range(1, 11):
            results_file = f'{base_dir}/keras_{flanking_size}_{site}_{i}/mutagenesis_results.csv'
            df = pd.read_csv(results_file)
            dataframes.append(df)
        results_df = pd.concat(dataframes, ignore_index=True)
        print(results_df.head())
        results_df.to_csv(f'{base_dir}/keras_{flanking_size}_{site}_results.csv', index=False)
    
        output_base = f'{base_dir}/keras_{flanking_size}_{site}'
        
        visualize(results_df, site, output_base)

if __name__ == '__main__':
    
    mutagenesis()