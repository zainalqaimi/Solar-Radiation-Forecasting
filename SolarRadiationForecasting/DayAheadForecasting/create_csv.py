import csv

# Specify your header names here
header = ['L', 'H', 'loss_function', 'batch_size', 'learning_rate', 'epochs', 'dropout', 'num_encoder_layers', 'num_decoder_layers', 'hidden_dim', 
          'encoder_output_dim', 'decoder_output_dim', 'temp_hidden_dim', 'train_loss_mse', 
          'val_loss_mse', 'test_loss_mse', 'train_loss_dtw', 
          'val_loss_dtw', 'test_loss_dtw']

# Open the file in write mode ('w') and create a writer object
with open('experiments.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    # Write the header
    writer.writerow(header)