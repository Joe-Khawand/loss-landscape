import os

if __name__=='__main__':
    """Runs the full experiment comparing multiple neural network optimizers
    """
    
    optims=['sgd','adam','natgrad']

    for opt in optims:
        print("Running ",opt," experiment ")
        command=(
            "python3 train.py "
            "--save_folder_name resnet20 "
            "--model resnet20"
            f"--optim {opt}"
            "--plot"
            "--plot_res 300"
            "--plot_size 300"
        )
        os.system(command=command)