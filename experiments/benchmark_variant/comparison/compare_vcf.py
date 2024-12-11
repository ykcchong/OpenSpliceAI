import pysam
import pandas as pd
import numpy as np

def parse_spliceai_info(info_field):
    """
    Parses the SpliceAI INFO field and returns a list of dictionaries with the scores.
    """
    spliceai_data = info_field.get('SpliceAI', None)
    if not spliceai_data:
        return []
    results = []
    # SpliceAI field can have multiple annotations separated by commas
    annotations = spliceai_data
    for ann in annotations:
        fields = ann.split('|')
        if len(fields) < 6:
            continue
        gene, var_type, ds_ag, ds_al, ds_dg, ds_dl = fields[:6]
        if ds_ag == '.' or ds_al == '.' or ds_dg == '.' or ds_dl == '.':
            continue
        max_ds = max(map(float, [ds_ag, ds_al, ds_dg, ds_dl]))
        result = {
            'Gene': gene,
            'DS_AG': float(ds_ag),
            'DS_AL': float(ds_al),
            'DS_DG': float(ds_dg),
            'DS_DL': float(ds_dl),
            'Max_DS': max_ds
        }
        results.append(result)
    return results

def vcf_to_dataframe(vcf_file):
    """
    Reads a VCF file and extracts variant positions and SpliceAI delta scores.
    Returns a pandas DataFrame.
    """
    variants = []
    vcf = pysam.VariantFile(vcf_file)
    for record in vcf:
        # Get basic variant info
        chrom = record.chrom
        pos = record.pos
        ref = record.ref
        alts = record.alts  # Can be multiple alternate alleles
        # For each ALT allele, get SpliceAI scores
        for i, alt in enumerate(alts):
            key = f'{chrom}_{pos}_{ref}_{alt}'
            info_field = record.info
            spliceai_scores = parse_spliceai_info(info_field)
            if spliceai_scores:
                for score in spliceai_scores:
                    variant_info = {
                        'Chromosome': chrom,
                        'Position': pos,
                        'Ref': ref,
                        'Alt': alt,
                        'Key': key
                    }
                    variant_info.update(score)
                    variants.append(variant_info)
            else:
                # No SpliceAI data for this variant
                variant_info = {
                    'Chromosome': chrom,
                    'Position': pos,
                    'Ref': ref,
                    'Alt': alt,
                    'Key': key,
                    'Gene': None,
                    'DS_AG': pd.NA,
                    'DS_AL': pd.NA,
                    'DS_DG': pd.NA,
                    'DS_DL': pd.NA,
                    'Max_DS': pd.NA
                }
                variants.append(variant_info)
    df = pd.DataFrame(variants)
    return df

def compare_variants(df1, df2, score_threshold, diff_threshold):
    """
    Merges two DataFrames on variant keys and compares SpliceAI delta scores.
    Excludes variants where all SpliceAI scores are NaN in both DataFrames.
    Returns a DataFrame with comparison results.
    """
    merged_df = pd.merge(df1, df2, on=['Chromosome', 'Position', 'Ref', 'Alt'], suffixes=('_1', '_2'), how='inner')

    # List of SpliceAI score columns in both DataFrames
    spliceai_cols_1 = ['DS_AG_1', 'DS_AL_1', 'DS_DG_1', 'DS_DL_1', 'Max_DS_1']
    spliceai_cols_2 = ['DS_AG_2', 'DS_AL_2', 'DS_DG_2', 'DS_DL_2', 'Max_DS_2']

    # Create a mask for rows where all SpliceAI scores are NaN in both datasets
    mask_all_nan_1 = merged_df[spliceai_cols_1].isna().all(axis=1)
    mask_all_nan_2 = merged_df[spliceai_cols_2].isna().all(axis=1)
    mask_all_nan_both = mask_all_nan_1 & mask_all_nan_2

    # Remove rows where all SpliceAI scores are NaN in both datasets
    merged_df = merged_df[~mask_all_nan_both]
    
    print('Percentage of variants above score threshold in Keras', merged_df['Max_DS_1'].ge(score_threshold).sum() / len(merged_df) * 100)
    print('Percentage of variants above score threshold in PyTorch', merged_df['Max_DS_2'].ge(score_threshold).sum() / len(merged_df) * 100)

    # Proceed with comparison
    # Calculate the absolute difference in Max_DS, handling NaN values
    merged_df['Delta_Max_DS'] = abs(merged_df['Max_DS_1'] - merged_df['Max_DS_2'])

    # Determine if the variants are similar based on the score threshold
    # Variants with NaN in Max_DS will not be marked as Similar
    count = np.where((merged_df['Max_DS_1'] >= score_threshold) & (merged_df['Max_DS_2'] >= score_threshold), True, False)
    similar = merged_df['Delta_Max_DS'] <= diff_threshold
    
    merged_df['Similar'] = similar & count
    merged_df['Count'] = count

    return merged_df

def main(vcf_file1, vcf_file2, score_threshold, diff_threshold, output_base):
    df1 = vcf_to_dataframe(vcf_file1)
    df2 = vcf_to_dataframe(vcf_file2)
    print('Keras', df1)
    print('PyTorch', df2)
    
    comparison_df = compare_variants(df1, df2, score_threshold, diff_threshold)
    print('Merged', comparison_df)
    
    # Print summary statistics
    total_variants = comparison_df['Count'].sum()
    similar_variants = comparison_df['Similar'].sum()
    similarity_percentage = (similar_variants / total_variants) * 100 if total_variants > 0 else 0
    print(f'Total variants compared: {total_variants}')
    print(f'Variants with delta score difference <= {score_threshold}: {similar_variants}')
    print(f'Similarity percentage: {similarity_percentage:.2f}%')
    # Save the comparison DataFrame to a CSV file
    comparison_df.to_csv(f'{output_base}/variant_comparison.csv', index=False)
    print('Comparison results saved to variant_comparison.csv')

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser(description='Compare two VCF files based on SpliceAI delta scores.')
    # parser.add_argument('vcf_file1', help='First VCF file')
    # parser.add_argument('vcf_file2', help='Second VCF file')
    # parser.add_argument('score_threshold', type=float, help='Score threshold for similarity')
    # args = parser.parse_args()   
    # main(args.vcf_file1, args.vcf_file2, args.score_threshold)
    
    vcf_keras = '/ccb/cybertron/smao10/openspliceai/experiments/benchmark_variant/comparison/keras/result.vcf'
    vcf_pytorch = '/ccb/cybertron/smao10/openspliceai/experiments/benchmark_variant/comparison/pytorch/result.vcf'
    score_threshold = 0.01
    diff_threshold = 0.01
    output_base = '/ccb/cybertron/smao10/openspliceai/experiments/benchmark_variant/comparison'
    main(vcf_keras, vcf_pytorch, score_threshold, diff_threshold, output_base)