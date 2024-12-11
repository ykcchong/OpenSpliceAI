# Description: Visualize the results of the mutagenesis experiment for the splice site prediction model.
import pandas as pd
import numpy as np
from spliceaitoolkit.constants import *
import logomaker
import matplotlib.pyplot as plt

import itertools

        
##############################################
## UTILS: ONE-HOT ENCODING, MUTATION, LOGOS
##############################################

# Function to mutate a base to the three other bases
def mutate_base(base):
    bases = ['A', 'C', 'G', 'T']
    return [b for b in bases if b != base]

# Function to calculate average score change
def calculate_average_score_change(ref_scores, mut_scores):
    return ref_scores - np.mean(mut_scores, axis=0)

# Function to generate DNA logo
def generate_dna_logo(score_changes, output_file, start=140, end=260):
    order = ['A', 'C', 'G', 'T']
    
    data_df = pd.DataFrame(score_changes, columns=['A', 'C', 'G', 'T']).astype(float)
    # Ensure valid start and end range
    if start < 0 or end > len(data_df):
        raise ValueError("Invalid start or end range for the given data.")
    # Fill any missing values with 0, just in case
    data_df = data_df.fillna(0)
    # Slice the DataFrame to include only rows from start to end
    data_df = data_df.iloc[start:end]
    print(data_df)
    logo = logomaker.Logo(data_df, stack_order='small_on_top', color_scheme='classic')
    logo.ax.set_title('DNA Logo - Score Change by Base')
    plt.savefig(output_file)

# Function to generate line plot for average score change
def plot_average_score_change(average_score_change, output_file, start=0, end=400):
    # Ensure valid start and end range
    if start < 0 or end > len(average_score_change):
        raise ValueError("Invalid start or end range for the given data.")
    
    # Slice the series/dataframe to include only rows from start to end
    sliced_average_change = average_score_change.iloc[start:end]
    
    plt.figure()
    plt.plot(sliced_average_change, label="Average Score Change")
    plt.title("Average Score Change by Position")
    plt.xlabel("Position")
    plt.ylabel("Score Change")
    plt.legend()
    plt.savefig(output_file)
    
    
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
    generate_dna_logo(pivot_df, f'{output_base}_dna_logo.png')
    print('saved to', f'{output_base}_dna_logo.png')

    ### Calculate average score change for each base and plot
    # acceptor_score_change = acceptor_avg_df.apply(lambda row: row['ref'] - np.mean([row['A'], row['C'], row['G'], row['T']]), axis=1)
    # donor_score_change = donor_avg_df.apply(lambda row: row['ref'] - np.mean([row['A'], row['C'], row['G'], row['T']]), axis=1)

    # if site == 'acceptor':
    #     plot_average_score_change(acceptor_score_change, f'{output_dir}/acceptor_average_score_change.png')
    # else:
    #     plot_average_score_change(donor_score_change, f'{output_dir}/donor_average_score_change.png')

def mutagenesis():
        
    sites = ['donor', 'acceptor']
    # flanking_sizes = [80, 400, 2000, 10000]
    flanking_sizes = [10000]
    exp_number = 7
    
    base_dir = f'/ccb/cybertron/smao10/openspliceai/experiments/mutagenesis/experiment_2/results/exp_{exp_number}/keras_job'
    
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