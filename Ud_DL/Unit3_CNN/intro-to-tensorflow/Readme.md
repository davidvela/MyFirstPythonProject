# TensorFlow Neural Network Lab

## initialization
create a new conda environment 

### WINDOWS 
conda env create -f environment_win.yml
This will create an environment called dlnd-tf-lab. You can enter the environment with the command
activate dlnd-tf-lab

conda install --yes --file requirements.txt


### MAC/LINUX 
conda env create -f environment.yml
This will create an environment called dlnd-tf-lab. You can enter the environment with the command
source activate dlnd-tf-lab

## 3 problems for you to solve:
* Problem 1: Normalize the features
* Problem 2: Use TensorFlow operations to create features, labels, weight, and biases tensors
* Problem 3: Tune the learning rate, number of steps, and batch size for the best accuracy