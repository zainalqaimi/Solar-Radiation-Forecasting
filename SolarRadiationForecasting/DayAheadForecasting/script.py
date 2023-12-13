from itertools import product
import subprocess
# from ray import tune
# from ray.tune.schedulers import ASHAScheduler


# Script for running all experiments
Ls = [192]
H = 96
loss = 'mse'
epochs = 10

# Experiment on GPU 1
batch_sizes = [128]
lrs = [0.0001]
dropouts = [0.0]

num_encoder_layers = [1]
num_decoder_layers = [1]
hidden_dims = [16]
enc_output_dims = [16]
decoder_output_dims = [4]
temp_hidden_dims = [16]

# Product of all hyperparameters
all_params = product(Ls, temp_hidden_dims, decoder_output_dims, enc_output_dims, hidden_dims, num_decoder_layers, num_encoder_layers, dropouts, lrs, batch_sizes)
# all_params = product(batch_sizes, lrs, dropouts)

for params in all_params:
    L, temp_hidden_dim, decoder_output_dim, enc_output_dim, hidden_dim, num_decoder_layer, num_encoder_layer, dropout, lr, batch_size = params
    # batch_size, lr, dropout = params
    command = (
        f"python main.py "
        f"--is_training 1 --model NoWeather_MLPCopula --L {L} --H {H} "
        f"--train_epochs {epochs} "
        f"--batch_size {batch_size} "
        f"--learning_rate {lr} "
        f"--dropout {dropout} "
        f"--num_enc_layers {num_encoder_layer} "
        f"--num_dec_layers {num_encoder_layer} "
        f"--hidden_dim {hidden_dim} "
        f"--enc_output_dim {enc_output_dim} "
        f"--dec_output_dim {decoder_output_dim} "
        f"--temporal_hidden_dim {hidden_dim} "
        f"--loss {loss} "
    )
    # command = (
    #     f"python main.py "
    #     f"--is_training 1 --model TiDEFolsom --L {L} --H {H} "
    #     f"--train_epochs {epochs} "
    #     f"--batch_size {batch_size} "
    #     f"--learning_rate {lr} "
    #     f"--dropout {dropout} "
    #     f"--loss {loss} "
    # )
    print("Running hyperparameters:", command)

    
    process = subprocess.Popen(command, shell=True)
    process.wait()



