import pysam 
from spliceaitoolkit.variant.utils import *

# vcf = pysam.VariantFile('test.vcf')

# import torch
# from collections import OrderedDict

# # Load the state_dict
# state_dict = torch.load('models/spliceai-mane/400nt/model_400nt_rs42.pt')

# # Print the keys in the state_dict
# print("State_dict keys:")
# for key in state_dict.keys():
#     print(key)

# # Assume `YourModelClass` is your model class
# device = setup_device()
# model, params = load_model(device, 400)

# # Print model's named parameters
# print("\nModel's named parameters:")
# for name, param in model.named_parameters():
#     print(name, param)

# # Optionally, load state_dict with strict=False
# try:
#     model.load_state_dict(state_dict, strict=False)
#     print("Model loaded successfully with strict=False.")
# except Exception as e:
#     print(f"Error loading model with strict=False: {e}")