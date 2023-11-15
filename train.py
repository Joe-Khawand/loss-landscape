import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import os
from tqdm import tqdm
from cifar10.model_loader import load

from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training models on cifar')
    parser.add_argument('--save_folder_name',type=str,help='Save folder for the models')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--model', default='vgg9', help='vgg9 | resnet20 ...')
    parser.add_argument('--epochs', default=30, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.01, type=int, help='learning rate')
    parser.add_argument('--nb_workers',default=4,type=int,help='Number of threads for the dataloader')
    parser.add_argument('--optim',default='sgd',type=str,help='Choose optimizer sgd | adam | natgrad')
    parser.add_argument('--weight_decay',default=0,type=int,help='weight decay')
    parser.add_argument('--betas',default=(0.9,0.999),help= 'betas for adam optim' )
    parser.add_argument('--momentum',default=0.9,type=int,help='Momentum for sgd')
    parser.add_argument('--plot', action='store_true', default=False, help='plot pca trajectory after training')
    parser.add_argument('--plot_res',type=int,default=100,help='Plot resulotion')
    parser.add_argument('--plot_size',type=int,default=100,help='Plot size')

    args = parser.parse_args()

    assert args.save_folder_name!=None,'Please specify a folder to save the trained models'

    model_folder_path='./cifar10/trained_nets/'
    
    assert not os.path.exists(model_folder_path+args.save_folder_name),'Folder already exists please specify another folder'

    os.mkdir(path=model_folder_path+args.save_folder_name)

    print('---------------')
    print('Training ',args.model,' for ',args.epochs, ' with a learning rate of ',args.lr, ' on Cifar-10')
    print('---------------\n')

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CIFAR-10 dataset
    
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    
    transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    
    cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


    train_loader = DataLoader(cifar_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nb_workers)

    # Initialize the network, loss function, and optimizer
    net = load(args.model, model_file=None, data_parallel=False)
    net.to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optim=='sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim=='adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr,betas=args.betas, weight_decay=args.weight_decay)
    elif args.optim=='natgrad':
        optimizer = optim.SGD(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError('Optimizer not implemented')
    
    max_epochs = args.epochs

    # Add Reduce on Plateau scheduler
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training loop
    for epoch in range(max_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        net.train()
        for data in tqdm(train_loader,desc='Epoch '+str(epoch+1)):
            inputs, labels = data

            inputs=inputs.to(device)
            labels=labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        
        training_accuracy = 100.0 * correct / total
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Training Accuracy: {training_accuracy}%")

        # Adjust the learning rate using the scheduler
        #scheduler.step(running_loss)

        # Save the model at this epoch
        model_filename = f"./cifar10/trained_nets/{args.save_folder_name}/model_{epoch+1}.t7"
        torch.save(net.state_dict(), model_filename)

    print("Finished Training")

    if args.plot:
        command = (
            "python3 pca_contour_trajectory.py "
            "--cuda "
            f"--model {args.model} "
            f"--x=-{args.plot_size}:{args.plot_size}:{args.plot_res} "
            f"--y=-{args.plot_size}:{args.plot_size}:{args.plot_res} "
            "--xnorm filter --xignore biasbn --ynorm filter --yignore biasbn "
            "--threads 10 "
            f"--model_file cifar10/trained_nets/{args.save_folder_name}/model_{max_epochs}.t7 "
            f"--model_folder cifar10/trained_nets/{args.save_folder_name}/ "
            "--start_epoch 1 "
            f"--max_epoch {max_epochs} "
            "--plot"
        )
        os.system(command=command)
