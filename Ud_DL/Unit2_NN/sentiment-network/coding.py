from collections import Counter
import time
import sys
import numpy as np
import matplotlib.colors as colors

g = open('./data/sent_an/reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('./data/sent_an/labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()
total_counts = Counter()

def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")

def printTest(): 
    len(reviews)   # 25000 
    print("labels.txt \t : \t reviews.txt\n")
    pretty_print_review_and_label(2137)
    pretty_print_review_and_label(12816)
    pretty_print_review_and_label(6267)
    pretty_print_review_and_label(21934)
    pretty_print_review_and_label(5297)
    pretty_print_review_and_label(4998)
#
def theory_validation():
    positive_counts = Counter()
    negative_counts = Counter()
    length = len(reviews)
    for  i in range(length): 
        if(i%5000 == 0):
            print( str(i) + 'out of' + str(length))
        if(labels[i] == 'POSITIVE'):
            for word in reviews[i].split(' '): 
                positive_counts[word] += 1.
                total_counts[word] += 1.
        else:
            for word in reviews[i].split(' '): 
                negative_counts[word] += 1.
                total_counts[word] += 1.

    print('finish');   print('posives')
    print(positive_counts.most_common(10))
    print('negatives')
    print(negative_counts.most_common(10))


    # TODO: Calculate the ratios of positive and negative uses of the most common words
    #       Consider words to be "common" if they've been used at least 100 times
    pos_neg_ratios = Counter()
    for word,cnt in list(total_counts.most_common()): 
        if(cnt > 100):
            pos_neg_ratio = positive_counts[word] / float(negative_counts[word]+1)
            pos_neg_ratios[word] = pos_neg_ratio
    print('finish')
    print(positive_counts["amazing"]) #1058.0
    print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
    print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
    print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))
    # TODO: Convert ratios to logs
    pos_neg_ratios_o = pos_neg_ratios
    for word,ration in list(pos_neg_ratios_o.most_common()):
        pos_neg_ratios[word] = np.log(pos_neg_ratios[word])
    print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
    print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
    print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))
    pos_neg_ratios.most_common(10)
    list(reversed(pos_neg_ratios.most_common()))[0:10]

#
def words2num():
    # TODO: Create set named "vocab" containing all of the words from all of the reviews
    vocab = set(total_counts)
    layer_0 =  np.zeros((1,vocab_size) ) #(1, 74074)
    # Create a dictionary of words in the vocabulary mapped to index positions
    # (to be used in layer_0)
    word2index = {}
    for i,word in enumerate(vocab):
        word2index[word] = i
        
    # display the map of words to indices
    len(word2index)
    word2index;
    def get_target_for_label(label):
        if label == 'POSITIVE':
            return 1
        else :
            return 0
    def update_input_layer(review):
        global layer_0
        layer_0 *= 0
        for word in  review.split(' '):
            layer_0[0][word2index[word]] += 1
    update_input_layer(reviews[0])
    get_target_for_label(labels[0])

#
def build_NN():
    class SentimentNetwork:
        def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
            # Assign a seed to our random number generator to ensure we get
            # reproducable results during development 
            np.random.seed(1)
            # process the reviews and their associated labels so that everything
            # is ready for training
            self.pre_process_data(reviews, labels)
            # Build the network to have the number of hidden nodes and the learning rate that
            # were passed into this initializer. Make the same number of input nodes as
            # there are vocabulary words and create a single output node.
            self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

        def pre_process_data(self, reviews, labels):  
            review_vocab = set()      
            for review in reviews:
                for word in review.split(" "):
                    review_vocab.add(word)
            # Convert the vocabulary set to a list so we can access words via indices
            self.review_vocab = list(review_vocab)
            label_vocab = set(labels)
            self.label_vocab = list(label_vocab)
            self.review_vocab_size = len(self.review_vocab)
            self.label_vocab_size = len(self.label_vocab)
            self.word2index = {}
            for i, word in enumerate(self.review_vocab):
                self.word2index[word] = i

            self.label2index = {}
            for i, label in enumerate(self.label_vocab):
                self.label2index[label] = i
            
        def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
            # Store the number of nodes in input, hidden, and output layers.
            self.input_nodes = input_nodes
            self.hidden_nodes = hidden_nodes
            self.output_nodes = output_nodes
            self.learning_rate = learning_rate
            self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
            self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                    (self.hidden_nodes, self.output_nodes))
            self.layer_0 = np.zeros((1,input_nodes))
        
            
        def update_input_layer(self,review):
            self.layer_0 *= 0
            for word in  review.split(' '):
                if( word in self.word2index.keys()):
                    self.layer_0[0][self.word2index[word]] += 1
                    
        def get_target_for_label(self,label):
            if label == 'POSITIVE':
                return 1
            else :
                return 0
            
        def sigmoid(self,x):
            return 1 / (1 + np.exp(-x))

        
        def sigmoid_output_2_derivative(self,output):
            return output * (1 - output)
        def train(self, training_reviews, training_labels):
            assert(len(training_reviews) == len(training_labels))
            correct_so_far = 0
            start = time.time()
            for i in range(len(training_reviews)):
                review = training_reviews[i] 
                label  = training_labels[i] 
                self.update_input_layer(review)
                layer_1 = self.layer_0.dot(self.weights_0_1)
                layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))        
                target = self.get_target_for_label(label)
            
                layer_2_error = layer_2 - target
                layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)
                layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
                layer_1_delta = layer_1_error
            
                self.weights_1_2 -= layer_1.T.dot(layer_2_delta) * self.learning_rate
                self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate 
            
                if(layer_2 >= 0.5 and label == 'POSITIVE'):
                    correct_so_far += 1
                elif(layer_2 < 0.5 and label == 'NEGATIVE'):
                    correct_so_far += 1
                elapsed_time = float(time.time() - start)
                reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
                
                sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4]                              + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5]                              + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1)                              + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
                if(i % 2500 == 0):
                    print("")
    
        def test(self, testing_reviews, testing_labels):
            correct = 0
            start = time.time()
            for i in range(len(testing_reviews)):
                pred = self.run(testing_reviews[i])
                if(pred == testing_labels[i]):
                    correct += 1
                elapsed_time = float(time.time() - start)
                reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
                sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4]                              + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5]                              + " #Correct:" + str(correct) + " #Tested:" + str(i+1)                              + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
        def run(self, review):
            self.update_input_layer(review.lower())
            layer_1 = self.layer_0.dot(self.weights_0_1)
            layer_2 = self.sigmoid( layer_1.dot(self.weights_1_2))      
            if(layer_2 >= 0.5):
                return 'POSITIVE'
            elif(layer_2 < 0.5) :
                return 'NEGATIVE'

        
    # mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)
    mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.01)
    mlp.train(reviews[:-1000],labels[:-1000])
    mlp.test(reviews[-1000:],labels[-1000:])

#
def reduce_noise(): 
    pass 
#
def test_efficiency_concept():
    layer_0 = np.zeros(10)
    layer_0[4] = 1
    layer_0[9] = 1
    weights_0_1 = np.random.randn(10,5) 
    layer_1 = np.zeros(5)

    layer_1 = layer_0.dot(weights_0_1) # SAME
    print(layer_1)

    layer_1 *= 0
    indices = [4,9]
    for index in indices:
        layer_1 += (1 * weights_0_1[index])
    print(layer_1) # SAME

 
#
def NN_moreEfficient():
    class SentimentNetwork:
        def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1, min_count = 10,polarity_cutoff = 0.1):
            np.random.seed(1)
            self.pre_process_data(reviews, labels, min_count, polarity_cutoff)
            self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

        def pre_process_data(self, reviews, labels, min_count, polarity_cutoff):
            
            positive_counts = Counter()
            negative_counts = Counter()
            total_counts = Counter()
            
            for i in range(len(reviews)):
                if(labels[i] == 'POSITIVE'):
                    for word in reviews[i].split(" "):
                        positive_counts[word] += 1
                        total_counts[word] += 1
                else:
                    for word in reviews[i].split(" "):
                        negative_counts[word] += 1
                        total_counts[word] += 1

            pos_neg_ratios = Counter()

            for term,cnt in list(total_counts.most_common()):
                if(cnt >= 50):
                    pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
                    pos_neg_ratios[term] = pos_neg_ratio

            for word,ratio in pos_neg_ratios.most_common():
                if(ratio > 1):
                    pos_neg_ratios[word] = np.log(ratio)
                else:
                    pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))
            
            
            review_vocab = set()
            for review in reviews:
                for word in review.split(" "):
                    ## New for Project 6: only add words that occur at least min_count times
                    #                     and for words with pos/neg ratios, only add words
                    #                     that meet the polarity_cutoff
                    if(total_counts[word] > min_count):
                        if(word in pos_neg_ratios.keys()):
                            if((pos_neg_ratios[word] >= polarity_cutoff) or (pos_neg_ratios[word] <= -polarity_cutoff)):
                                review_vocab.add(word)
                        else:
                            review_vocab.add(word)
            self.review_vocab = list(review_vocab)
        
            label_vocab = set(labels)
            self.label_vocab = list(label_vocab)
            self.review_vocab_size = len(self.review_vocab)
            self.label_vocab_size = len(self.label_vocab)
            self.word2index = {}
            for i, word in enumerate(self.review_vocab):
                self.word2index[word] = i
            self.label2index = {}
            for i, label in enumerate(self.label_vocab):
                self.label2index[label] = i
            
        def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
            # Store the number of nodes in input, hidden, and output layers.
            self.input_nodes = input_nodes
            self.hidden_nodes = hidden_nodes
            self.output_nodes = output_nodes
            self.learning_rate = learning_rate
            self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
            self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                    (self.hidden_nodes, self.output_nodes))
            self.layer_1 = np.zeros((1,hidden_nodes))                
        def get_target_for_label(self,label):
            if label == 'POSITIVE':
                return 1
            else :
                return 0
            
        def sigmoid(self,x):
            return 1 / (1 + np.exp(-x))   
        def sigmoid_output_2_derivative(self,output):
            return output * (1 - output)

        def get_indices(self, review):
            indices = set()
            for word in review.split(" "):
                if(word in self.word2index.keys()):
                    indices.add(self.word2index[word])
            return indices
    
        def train(self, training_reviews_raw, training_labels):       
            training_reviews = list()
            for review in training_reviews_raw:
                indices = self.get_indices(review)
                training_reviews.append(list(indices))
            assert(len(training_reviews) == len(training_labels))
            correct_so_far = 0
            start = time.time()
            for i in range(len(training_reviews)):
                review = training_reviews[i] 
                label  = training_labels[i] 
                # self.update_input_layer(review)
                # layer_1 = self.layer_0.dot(self.weights_0_1)
                self.layer_1 *= 0
                for index in review:
                    self.layer_1 += self.weights_0_1[index]
                layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
                target = self.get_target_for_label(label)
    
                layer_2_error = layer_2 - target
                layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)       
                layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
                layer_1_delta = layer_1_error
                self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate            
                # self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate 
                for index in review:
                    self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate
                if(layer_2 >= 0.5 and label == 'POSITIVE'):
                    correct_so_far += 1
                elif(layer_2 < 0.5 and label == 'NEGATIVE'):
                    correct_so_far += 1
                elapsed_time = float(time.time() - start)
                reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
                
                sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                                + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                                + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                                + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
                if(i % 2500 == 0):
                    print("")
    
        def test(self, testing_reviews, testing_labels):
            correct = 0
            start = time.time()
            for i in range(len(testing_reviews)):
                pred = self.run(testing_reviews[i])
                if(pred == testing_labels[i]):
                    correct += 1
                elapsed_time = float(time.time() - start)
                reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
                
                sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                                + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                                + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                                + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
        def run(self, review)
            indices = self.get_indices(review.lower())
            self.layer_1 *= 0
            for index in indices:
                self.layer_1 += (self.weights_0_1[index])
            
            layer_2 = self.sigmoid( self.layer_1.dot(self.weights_1_2))
            if(layer_2 >= 0.5):
                return 'POSITIVE'
            elif(layer_2 < 0.5) :
                return 'NEGATIVE'

    #mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.05,learning_rate=0.01)
    mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.8,learning_rate=0.01)
    mlp.train(reviews[:-1000],labels[:-1000])
    mlp.test(reviews[-1000:],labels[-1000:])

    mlp_full = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=0,polarity_cutoff=0,learning_rate=0.01)
    mlp_full.train(reviews[:-1000],labels[:-1000])
    def get_most_similar_words(focus = "horrible"):
        most_similar = Counter()
        for word in mlp_full.word2index.keys():
            most_similar[word] = np.dot(mlp_full.weights_0_1[mlp_full.word2index[word]],mlp_full.weights_0_1[mlp_full.word2index[focus]])
        return most_similar.most_common()
    get_most_similar_words("excellent")
    get_most_similar_words("terrible")
    def visualize_words():
        words_to_visualize = list()
        for word, ratio in pos_neg_ratios.most_common(500):
            if(word in mlp_full.word2index.keys()):
                words_to_visualize.append(word)
            
        for word, ratio in list(reversed(pos_neg_ratios.most_common()))[0:500]:
            if(word in mlp_full.word2index.keys()):
                words_to_visualize.append(word)
        pos = 0
        neg = 0
        colors_list = list()
        vectors_list = list()
        for word in words_to_visualize:
            if word in pos_neg_ratios.keys():
                vectors_list.append(mlp_full.weights_0_1[mlp_full.word2index[word]])
                if(pos_neg_ratios[word] > 0):
                    pos+=1
                    colors_list.append("#00ff00")
                else:
                    neg+=1
                    colors_list.append("#000000")
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=0)
        words_top_ted_tsne = tsne.fit_transform(vectors_list)
        p = figure(tools="pan,wheel_zoom,reset,save",
                toolbar_location="above",
                title="vector T-SNE for most polarized words")

        source = ColumnDataSource(data=dict(x1=words_top_ted_tsne[:,0],
                                            x2=words_top_ted_tsne[:,1],
                                            names=words_to_visualize,
                                            color=colors_list))

        p.scatter(x="x1", y="x2", size=8, source=source, fill_color="color")

        word_labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,
                        text_font_size="8pt", text_color="#555555",
                        source=source, text_align='center')
        p.add_layout(word_labels)
        show(p)
#      
if __name__ == "__main__":
    # P0 printTest( )
    # P1 theory_validation()
    # P2 words2num()
    # P3 build_NN()
    # P4 reduce_noise() use the post_ratio - pass - 
    # P5 indexes test_efficiency_concept()
    # P6 reducing more noise by trategically reducing the vocabullary! 
    NN_moreEfficient()