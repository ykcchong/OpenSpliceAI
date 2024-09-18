import time
from openspliceai.train_base.spliceai import *
from openspliceai.train_base.utils import *
from openspliceai.calibrate.temperature_scaling import *
from matplotlib import pyplot as plt

def initialize_model_and_optim(device, flanking_size, pretrained_model):
    L = 32
    N_GPUS = 2
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    BATCH_SIZE = 18 * N_GPUS
    if int(flanking_size) == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        BATCH_SIZE = 18 * N_GPUS
    elif int(flanking_size) == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        BATCH_SIZE = 18 * N_GPUS
    elif int(flanking_size) == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10])
        BATCH_SIZE = 12 * N_GPUS
    elif int(flanking_size) == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10, 25, 25, 25, 25])
        BATCH_SIZE = 6 * N_GPUS    
    CL = 2 * np.sum(AR * (W - 1))
    print("\033[1mContext nucleotides: %d\033[0m" % (CL))
    print("\033[1mSequence length (output): %d\033[0m" % (SL))
    # Initialize the model
    model = SpliceAI(L, W, AR).to(device)
    # Print the shapes of the parameters in the initialized model
    print("\nInitialized model parameter shapes:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}", end=", ")

    # Load the pretrained model
    state_dict = torch.load(pretrained_model, map_location=device)

    # Filter out unnecessary keys and load matching keys into model
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

    # Load state dict into the model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Print missing and unexpected keys
    print("\nMissing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    # Set up optimizer and scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    print(model, file=sys.stderr)    
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}
    return model, optimizer, scheduler, params



def calibrate(args):
    print('Running OpenSpliceAI with calibrate mode.')
    # assert training_target in ["RefSeq", "MANE", "SpliceAI", "SpliceAI27"]
    device = setup_environment(args)
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(args)
    train_h5f, test_h5f, batch_num = load_datasets(args)
    train_idxs, val_idxs, test_idxs = generate_indices(batch_num, args.random_seed, test_h5f)
    model, optimizer, scheduler, params = initialize_model_and_optim(device, args.flanking_size, args.pretrained_model)
    

    print("\n--------------------------------------------------------------")
    start_time = time.time()
    ##########################################
    # Temperature scaling
    ##########################################    
    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(model, test_h5f, test_idxs, device, params["BATCH_SIZE"], params, train=True)
    # scaled_model = ModelWithTemperature(model).to(device)
    # scaled_model.load_state_dict(torch.load("temp_model.pt"))
    # scaled_model = scaled_model.to(device)
    # scaled_model.set_temperature(model, train_h5f, val_idxs, device, params["BATCH_SIZE"], params, train=False)
    # print("logits: ", logits.shape)
    # print("labels: ", labels.shape)
    torch.save(scaled_model.state_dict(), f"temp_model.pt")
    print("scaled_model: ", scaled_model)
    end_time = time.time()
    print(f"Time taken to calibrate the model: {end_time - start_time:.2f} seconds")

    # Apply temperature scaling
    logits = scaled_model.logits
    labels = scaled_model.labels
    logits_scaled = None
    with torch.no_grad():
        logits_scaled = scaled_model.temperature_scale(logits)

    print("logits_scaled: ", logits_scaled)
    print("logits_scaled: ", logits_scaled.shape)
    print("logits: ", logits)
    print("logits: ", logits.shape)
    print("labels: ", labels)
    print("labels: ", labels.shape)

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=1)
    probs_scaled = torch.nn.functional.softmax(logits_scaled, dim=1)
    print("probs: ", probs)
    print("probs: ", probs.shape)
    print("probs_scaled: ", probs_scaled)
    print("probs_scaled: ", probs_scaled.shape)

    # Convert tensors to CPU before converting to NumPy for use with sklearn's log_loss
    probs = probs.detach().cpu().numpy()
    probs_scaled = probs_scaled.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    # # ##########################################
    # # # Plotting the score distribution
    # # ##########################################
    # # score_frequency_distribution(probs, probs_scaled, labels, index=0)
    # # score_frequency_distribution(probs, probs_scaled, labels, index=1)
    # # score_frequency_distribution(probs, probs_scaled, labels, index=2)
    # # Assuming labels are one-hot encoded, we convert them to class indices for log_loss
    # # Calculate log-loss
    # log_loss_before = log_loss(labels, probs)
    # log_loss_after = log_loss(labels, probs_scaled)
    # # print("probs_after_reshaped: ", probs_after_reshaped)
    # # print("labels_indices: ", labels_indices)
    # print("Log-loss of")
    # print(f" * uncalibrated classifier: {log_loss_before:.8f}")
    # print(f" * calibrated classifier: {log_loss_after:.8f}")


    ##########################################
    # Plotting the calibration curve
    ##########################################
    # Compute the calibration curve for each class 
    classes = ["Non-splice site", "Acceptor site", "Donor site"]
    from sklearn.calibration import calibration_curve 
    calibration_curve_values = [] 
    calibration_curve_scaled_values = [] 
    for i in range(3): 
        curve = calibration_curve(labels == i,  
                                probs[:, i],  
                                n_bins=30,  
                                pos_label=True)
        scaled_curve = calibration_curve(labels == i,  
                                probs_scaled[:, i],  
                                n_bins=30,  
                                pos_label=True)  
        calibration_curve_values.append(curve)
        calibration_curve_scaled_values.append(scaled_curve) 
    
    os.mkdir(f'{args.output_dir}/calibration')
    # Plot the calibration curves 
    fig, axs = plt.subplots(1, 3, figsize=(17,5)) 
    for i in range(3): 
        axs[i].plot(calibration_curve_values[i][1],  
                    calibration_curve_values[i][0],  
                    marker='o', label='Original probability') 
        axs[i].plot(calibration_curve_scaled_values[i][1],  
                    calibration_curve_scaled_values[i][0],  
                    marker='o', label='Calibrated probability') 
        axs[i].plot([0, 1], [0, 1], linestyle='--') 
        axs[i].set_xlim([0, 1]) 
        axs[i].set_ylim([0, 1]) 
        axs[i].set_title(f"{classes[i]}", fontsize = 17) 
        axs[i].set_xlabel("Predicted probability", fontsize = 15) 
        axs[i].set_ylabel("Empirical probability", fontsize = 15) 
        axs[i].legend()
    plt.tight_layout() 
    plt.savefig(f'{args.output_dir}/calibration/calibration_curve.png')
    # plt.savefig('calibration_curve.png')







    # params["RANDOM_SEED"] = args.random_seed
    # train_metric_files = create_metric_files(log_output_train_base)
    # valid_metric_files = create_metric_files(log_output_val_base)
    # test_metric_files = create_metric_files(log_output_test_base)
    # train_model(model, optimizer, scheduler, train_h5f, test_h5f, train_idxs, val_idxs, test_idxs, 
    #             model_output_base, args, device, params, train_metric_files, valid_metric_files, test_metric_files)
    train_h5f.close()
    test_h5f.close()
