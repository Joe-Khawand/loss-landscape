import os

if __name__=='__main__':
    """Runs the full experiment comparing multiple neural network optimizers
    """
    
    optims=['sgd','adam','natgrad']

    for opt in optims:
        print("\n--------------------------")
        print("Running ",opt," experiment ")
        print("--------------------------\n")
        command=(
            "python3 train.py "
            f"--save_folder_name resnet20_{opt} "
            "--model resnet20 "
            f"--optim {opt} "
            "--plot "
            "--plot_res 100 " 
            "--plot_size_x 500 --plot_size_y 300"
        )
        os.system(command=command)