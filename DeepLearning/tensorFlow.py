import numpy as np 
# user transactions in a bank  - 2 children and 3 accounts
# [2,3] => [5,1] => 9 

# forward propagation - multiply input for weights 
input_data = np.array([2,3])
weights = { 'node_0': np.array([1, 1]),
            'node_1': np.array([-1, 1]),
            'output': np.array([2, -1]) }
node_0_value = (input_data * weights['node_0']).sum()
node_1_value = (input_data * weights['node_1']).sum()

hidden_layer_values = np.array([node_0_value, node_1_value])
print(hidden_layer_values)

output = ( hidden_layer_values * weights['output']).sum()
print(output)

# Activation function! => capture Nonlinear Functions! 
#    popular before tahn(hidden_Nodes! )
node_0_output = np.tanh(node_0_value)
node_1_output = np.tanh(node_1_value)
hidden_layer_output = np.array([node_0_output, node_1_output])
output_tanh   = (hidden_layer_output * weights['output']).sum()
print(output_tanh)

#    popular now ReLU (Rectifier Linear Activation)   0 if x<0 && x if x>=0
#       Define ReLU acitvation function
def relu(input):
    output = max(0, input)
    return(output)

node_0_reulu = relu(node_0_value)
node_1_reulu = relu(node_1_value)


def predict_with_network(input_data_row, weights):
# ... 
    model_output = 'x'
    return()    

results = []
input_data = []
for input_data_row in input_data:
    results.append(predict_with_network(input_data_row, weights))

