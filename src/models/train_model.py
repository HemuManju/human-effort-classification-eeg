from visdom import Visdom
from utils import classification_accuracy, weights_init
import yaml
import os
import torch
from sklearn.model_selection import train_test_split
from networks import ShallowEEGNet

# Parameters
config = yaml.load(open(os.getcwd()+'/src/config.yml'))
epoch_length = config['epoch_length']
s_freq = config['s_freq']
n_electrodes = config['n_electrodes']

# Choose the device to train the network
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Computation device being used:', device)


def train(network, new_weights=False):
    """Main function to run the optimization.

    """
    # An instance of model
    model = network(OUTPUT).to(device)
    if new_weights:
        model.apply(weights_init)

    viz.close(win='Accuracy')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        for x_batch, y_batch in train_data_iterator:
            # Send the images and labels to gpu
            x_batch = x_batch.to(device)
            y_batch = (torch.max(y_batch, dim=1)[1]).to(device)

            # Forward pass
            out_put = model(x_batch)
            loss = criterion(out_put, y_batch)

            # Backward and optimize
            optimizer.zero_grad() # For batch gradient optimisation
            loss.backward()
            optimizer.step()


        # Visualization of accuarcy
        viz.line(X = np.column_stack([epoch]*3),
        Y = np.column_stack((train_accuracy, valid_accuracy, test_accuracy)),
        opts = opts, win='Accuracy', update='append')


if __name__=='__main__':
    train(ShallowEEGNet)
