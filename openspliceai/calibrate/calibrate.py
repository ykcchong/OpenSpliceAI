# calibrate.py
import os
import time
import torch
import numpy as np
from torch.nn import functional as F
from openspliceai.train_base.utils import *
from openspliceai.calibrate.calibrate_utils import *
from openspliceai.calibrate.visualization import *
from openspliceai.calibrate.temperature_scaling import *
from openspliceai.calibrate.model_utils import initialize_model_and_optim

def calibrate(args):
    print('Running OpenSpliceAI calibration')
    start_time = time.time()
    os.makedirs(f"{args.output_dir}/calibration", exist_ok=True)
    
    # Initialize model
    device = setup_environment(args)
    train_h5f, test_h5f, batch_num = load_datasets(args)
    train_idxs, val_idxs, test_idxs = generate_indices(batch_num, args.random_seed, test_h5f)
    model, params = initialize_model_and_optim(device, args.flanking_size, args.pretrained_model)
    scaled_model = ModelWithTemperature(model)
    # Check if temperature_file is provided
    valid_loader = get_validation_loader(test_h5f, test_idxs, params["BATCH_SIZE"])

    # Load/calculate temperature
    if args.temperature_file:
        scaled_model.load_temperature(args.temperature_file, valid_loader, params)
        print(f"Loaded temperature from {args.temperature_file}: {scaled_model.temperature.item()}")
        end_time = time.time()
        print(f"Time taken to load the calibrated model: {end_time - start_time:.2f} seconds")
    else:
        # Train the temperature
        scaled_model.set_temperature(valid_loader, params)
        # Save the temperature parameter
        temperature_save_path = f"{args.output_dir}/temperature.pt"
        scaled_model.save_temperature(temperature_save_path)
        print(f"Saved temperature to {temperature_save_path}")
        end_time = time.time()
        print(f"Time taken to calibrate the model: {end_time - start_time:.2f} seconds")

    temperature_txt_save_path = f"{args.output_dir}/temperature.txt"
    with open(temperature_txt_save_path, 'w') as f:
        f.write(f"{scaled_model.temperature.item()}\n")


    # Generate predictions
    with torch.no_grad():
        logits, labels = scaled_model.logits, scaled_model.labels
        logits_scaled = scaled_model.temperature_scale(logits)
    
    # Convert logits to probabilities
    probs = F.softmax(logits, 1).cpu().numpy()
    probs_scaled = F.softmax(logits_scaled, 1).cpu().numpy()
    labels = labels.cpu().numpy()
    
    ##########################################
    # Plotting the score distribution
    ##########################################
    for idx in [0, 1, 2]:
        plot_score_distribution(probs, probs_scaled, labels, args.output_dir, idx)
    
    calibration_data = []
    calibration_data_scaled = []
    for i in range(3):
        class_labels = (labels == i).astype(int)
        ct = compute_calibration_curve(class_labels, probs[:, i], 30, 'uniform')
        ct_scaled = compute_calibration_curve(class_labels, probs_scaled[:, i], 30, 'uniform')
        calibration_data.append(ct)
        calibration_data_scaled.append(ct_scaled)
        save_calibration_data(args.output_dir, ['Non-splice', 'Acceptor', 'Donor'][i], 
                             args.flanking_size, *ct, 'original')
        save_calibration_data(args.output_dir, ['Non-splice', 'Acceptor', 'Donor'][i], 
                             args.flanking_size, *ct_scaled, 'calibrated')
    
    plot_calibration_curves(calibration_data, calibration_data_scaled, 
                           ['Non-splice', 'Acceptor', 'Donor'], args.output_dir)
    
    brier_uncal, brier_cal = calculate_brier_scores(labels, probs, probs_scaled)
    plot_brier_scores(brier_uncal, brier_cal, ['Non-splice', 'Acceptor', 'Donor'], args.output_dir)
    
    plot_calibration_map(scaled_model, device, args.output_dir)
    
    print(f"Calibration completed in {time.time()-start_time:.2f}s")