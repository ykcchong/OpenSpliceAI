import logomaker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Function to mutate a base to the three other bases
def mutate_base(base):
    bases = ['A', 'C', 'G', 'T']
    return [b for b in bases if b != base]

# Function to calculate average score change
def calculate_average_score_change(ref_scores, mut_scores):
    return ref_scores - np.mean(mut_scores, axis=0)

# # Function to generate DNA logo
# def generate_dna_logo(score_changes, output_file, start=140, end=260):
#     order = ['A', 'C', 'G', 'T']
    
#     data_df = pd.DataFrame(score_changes, columns=['A', 'C', 'G', 'T']).astype(float)
#     # Ensure valid start and end range
#     if start < 0 or end > len(data_df):
#         raise ValueError("Invalid start or end range for the given data.")
#     # Fill any missing values with 0, just in case
#     data_df = data_df.fillna(0)
#     # Slice the DataFrame to include only rows from start to end
#     data_df = data_df.iloc[start:end]
#     print(data_df)
#     logo = logomaker.Logo(data_df, stack_order='small_on_top', color_scheme='classic')
#     logo.ax.set_title('DNA Logo - Score Change by Base')
#     plt.savefig(output_file)

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
    
# Function to generate DNA logo with gene annotation
def generate_dna_logo(score_changes, site, output_file, start=100, end=300):
    
    data_df = pd.DataFrame(score_changes, columns=['A', 'C', 'G', 'T']).astype(float)
    
    # Ensure valid start and end range
    if start < 0 or end > len(data_df):
        raise ValueError("Invalid start or end range for the given data.")
    
    midpoint = len(data_df) // 2
    if site == 'donor':
        midpoint -= 2
        exon_start, exon_stop = start, midpoint
    else:
        midpoint += 1
        exon_start, exon_stop = midpoint, end
        
    exon_start, exon_stop = exon_start-midpoint//2, exon_stop-midpoint//2
        
    # set parameters for drawing gene
    exon_start -= .5
    exon_stop += .5
    y = -.2
    xs = np.arange(start-midpoint, end+midpoint, len(data_df) // 10)
    ys = y*np.ones(len(xs))
    
    # Fill any missing values with 0, just in case
    data_df = data_df.fillna(0)
    # Slice the DataFrame to include only rows from start to end
    data_df = data_df.iloc[start:end].reset_index(drop=True)
    
    print(data_df)
    logo = logomaker.Logo(data_df, 
                          stack_order='fixed', 
                          color_scheme='classic', 
                          figsize=(25,8))
    
    # style using Logo methods
    
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left'], visible=True, bounds=[-0.5, 2.75])

    # style using Axes methods
    # Calculate the relative positions
    relative_positions = np.arange(start, end) - midpoint
    # logo.ax.set_xticks(range(len(relative_positions)))
    logo.ax.set_xticklabels(relative_positions, rotation=45, fontsize=10) # font increased
    
    logo.ax.set_xlim([-.5, data_df.shape[0]+.5])
    logo.ax.set_xticks([])
    logo.ax.set_ylim([-.5, 2.75])
    # logo.ax.set_yticks([0, 2.75])
    logo.ax.set_yticklabels(['0', '2.75'], fontsize=14) # font increased
    logo.ax.set_ylabel('             Score', labelpad=-1, fontsize=18) # font increased
    

    # draw gene
    logo.ax.axhline(y, color='k', linewidth=1)
    logo.ax.plot(xs, ys, marker='4', linewidth=0, markersize=7, color='k')
    logo.ax.plot([exon_start, exon_stop],
                    [y, y], color='k', linewidth=10, solid_capstyle='butt')

    # annotate gene
    
    # logo.ax.text(2,2*y,f'${site.capitalize()}$',fontsize=16) # font increased
    if site == 'donor':
        logo.ax.plot(exon_stop, 1.8*y, '^k', markersize=12) # marker decreased
        logo.ax.text(exon_stop, 2.5*y,f'{site.capitalize()}', verticalalignment='top', horizontalalignment='center', fontsize=12)
    else:
        logo.ax.plot(exon_start, 1.8*y, '^k', markersize=12) # marker decreased
        logo.ax.text(exon_start, 2.5*y,f'{site.capitalize()}', verticalalignment='top', horizontalalignment='center', fontsize=12) # font increased
    
    # logo.ax.set_title(f'OpenSpliceAI - {site.capitalize()} Site Score Change by Base')
    plt.savefig(output_file, bbox_inches='tight')