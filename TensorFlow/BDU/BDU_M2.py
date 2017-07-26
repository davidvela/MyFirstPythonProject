# TensorFlow - lesson 1 Big Data University
#----------------------------------------------
# MODULE 2 - convolutional networks 
# original goal of machine learning: 
#   "Move humananity closer to the undreachable General AI"

# Fully connected layer 
# - each neuron connected to every neuron in previuos layer
# - each connection has it's own weight 
# - no assumptions about the feautres in the input data 
# - type of layers are also very expensive in terms of memory and computation
#   
# Convolutional layer 
# - each neuron is only connected to a few nearby local neurons in previosu layer
# - same weights is used for every neuron 
# - this connection patterns only makes sense for cases where:
#   - data can be interpreted as spatial with the fatures to be extracetd being spatially local 
#       (e.g. a window consists of a set of pixels around an area, not spread all across the image)
#   - equally likely to occur at any input position (hence some weights at all positions)
#       (e.g. window can occur anywhere in the image and the image can be rotated or scaled)
#  
#
# FEATURE LEARNING  
#  Feature enginering is the process of extracting useful patterns from input data
#  that will help prediction model to understand better the real nature of the problem.  
#  CNN good for finding patterns - Feature Engineering 
#  Combining simple features to create bigger pictures s