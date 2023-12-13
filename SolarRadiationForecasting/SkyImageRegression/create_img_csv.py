import csv

# Specify your header names here
header = ['L', 'H', 'output_dim', 'hidden_dim', 'loss_function', 'batch_size', 'learning_rate', 'epochs', 'transform', 'train_loss_mse', 
          'val_loss_mse', 'test_loss_mse']

# Open the file in write mode ('w') and create a writer object
with open('img_experiments.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    # Write the header
    writer.writerow(header)