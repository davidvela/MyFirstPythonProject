# PART 4 Recurrent Neural Networks
Recurrent neural network is great for predicting on sequential data like music and text. With this neural network, you can generate new music, translate a language, or predict a seizure using an electroencephalogram. This section will teach you how to build and train a recurrent neural.
    Project: Generate TV Scripts
    Project: Language Translations

**Intro to RNN** 
Anna Karenina project - exercises!
- create get_batches! L6
- build network :8
- RNN output + loss L10,11
- Build network  


## Notes - 
### L8 test summarization 
summarize one text in one line using natural language processing... 
encoder/decoder and attention mechanism (prioritizer)
### L9 Sequence2Sequence 
download chatbot - standford chiphuyen/stanford-tensorflow-tutorials - understand code
+ https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html 

Reading the documentation for **tf.nn.dynamic_rnn** , you'll see tf.nn.dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None, dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope=None)

So you need to at least pass in the RNN cell you built (for example tf.contrib.rnn.BasicLSTMCell). You'll also need to give it the inputs tensor, which in this case is the input text data, typically coming from the embedding layer. I also typically pass in an initial_state which you've seen in the previous RNN lessons.
https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq 


tips and tricks from https://raw.githubusercontent.com/karpathy/char-rnn/
## Tips and Tricks

### Monitoring Validation Loss vs. Training Loss
If you're somewhat new to Machine Learning or Neural Networks it can take a bit of expertise to get good models. The most important quantity to keep track of is the difference between your training loss (printed during training) and the validation loss (printed once in a while when the RNN is run on the validation data (by default every 1000 iterations)). In particular:

- If your training loss is much lower than validation loss then this means the network might be **overfitting**. Solutions to this are to decrease your network size, or to increase dropout. For example you could try dropout of 0.5 and so on.
- If your training/validation loss are about equal then your model is **underfitting**. Increase the size of your model (either number of layers or the raw number of neurons per layer)

### Approximate number of parameters

The two most important parameters that control the model are `rnn_size` and `num_layers`. I would advise that you always use `num_layers` of either 2/3. The `rnn_size` can be adjusted based on how much data you have. The two important quantities to keep track of here are:

- The number of parameters in your model. This is printed when you start training.
- The size of your dataset. 1MB file is approximately 1 million characters.

These two should be about the same order of magnitude. It's a little tricky to tell. Here are some examples:

- I have a 100MB dataset and I'm using the default parameter settings (which currently print 150K parameters). My data size is significantly larger (100 mil >> 0.15 mil), so I expect to heavily underfit. I am thinking I can comfortably afford to make `rnn_size` larger.
- I have a 10MB dataset and running a 10 million parameter model. I'm slightly nervous and I'm carefully monitoring my validation loss. If it's larger than my training loss then I may want to try to increase dropout a bit and see if that heps the validation loss.

### Best models strategy

The winning strategy to obtaining very good models (if you have the compute time) is to always err on making the network larger (as large as you're willing to wait for it to compute) and then try different dropout values (between 0,1). Whatever model has the best validation performance (the loss, written in the checkpoint filename, low is good) is the one you should use in the end.

It is very common in deep learning to run many different models with many different hyperparameter settings, and in the end take whatever checkpoint gave the best validation performance.

By the way, the size of your training and validation splits are also parameters. Make sure you have a decent amount of data in your validation set or otherwise the validation performance will be noisy and not very informative.