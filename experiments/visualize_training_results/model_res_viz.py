import matplotlib.pyplot as plt
import os
import sys

# Function to load data from a file
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file.readlines()]
    return data

chunk_size = sys.argv[1]
flanking_size = sys.argv[2]
exp_num = sys.argv[3]
training_target = sys.argv[4]

assert training_target in ["MANE", "SpliceAI"]
# Directories
project_root = "/Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/"
base_dir = f"{project_root}results/model_train_outdir/train_test_dataset_{training_target}_splan_{chunk_size}chunk_{flanking_size}flank_spliceai_architecture/{exp_num}/LOG/"
viz_out_dir = f"{project_root}results/model_train_outdir/train_test_dataset_{training_target}_splan_{chunk_size}chunk_{flanking_size}flank_spliceai_architecture/{exp_num}/LOG/viz/"
os.makedirs(viz_out_dir, exist_ok=True)

phases = ['TRAIN', 'VAL', 'TEST']
metrics = ['accuracy', 'recall', 'f1', 'precision']#, 'topk']
categories = ['acceptor', 'donor', 'neither']

print(f"Visualizing training results from {base_dir}...")
print(f"Saving visualizations to {viz_out_dir}...")
# # Plotting
# plt.figure(figsize=(20, 10))
# for metric in metrics:
#     for category in categories:
#         plt.clf() # Clear current figure to avoid overlap
#         for phase in phases:
#             file_name = f'{category}_{metric}.txt'
#             file_path = os.path.join(base_dir, phase, file_name)
#             data = load_data(file_path)
#             plt.plot(data, label=f'{phase}')
#             plt.title(f'{category.capitalize()} {metric.capitalize()} over Steps')
#             plt.xlabel('Step')
#             plt.ylabel(metric.capitalize())
#             plt.legend()
#             plt.tight_layout()
#             plt.savefig(f'{viz_out_dir}{category}_{metric}_{phase}.png', dpi=300)
#             # plt.show()


# plt.clf() # Clear current figure to avoid overlap
# for phase in phases:
#     plt.clf() # Clear current figure to avoid overlap
#     file_name = f'loss.txt'
#     file_path = os.path.join(base_dir, phase, file_name)
#     data = load_data(file_path)
#     plt.plot(data, label=f'{phase}')
#     plt.title(f'{phase} Loss over Steps')
#     plt.xlabel('Step')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'{viz_out_dir}_Loss_{phase}.png', dpi=300)
#     # plt.show()