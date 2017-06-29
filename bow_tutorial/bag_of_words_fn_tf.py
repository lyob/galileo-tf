# From the text:
# We create a simple TensorFlow model function, that takes features (list of word IDs) and target (one of 15 classes). We use simple bow_encoder which combines creation of embedding matrix, lookup for each ID in the input and then averaging them. Then we just add a fully connected layer on top and the use it to compute loss and classification results tf.argmax(logits, 1). Adding training regime (Adam with 0.01 learning rate) and that's our function.

def bag_of_words_model(features, target):  
  """A bag-of-words model. Note it disregards the word order in the text."""  
  target = tf.one_hot(target, 15, 1, 0)  # this uses one-hot encoding for the targets (numbers 1 to 15 representing the categories)
  features = tf.contrib.layers.bow_encoder(      
      features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE) # while for the features it uses the 'bow_encoder' which acts in a similar way to Word2Vec. It maps each sentence to a fixed-size vector and then averages each vector. 
  logits = tf.contrib.layers.fully_connected(features, 15,
      activation_fn=None)  # lastly, we use a fully connected layer. This means that every node in this layer is connected to every node in the previous layer. It is trying to categorise the averaged vectors from the previous features fn. activation_fn default value is the ReLU function but setting it to None makes it linear.
  loss = tf.contrib.losses.softmax_cross_entropy(logits, target) # then we use cross-entropy matrix calculations to calculate the loss between the target and actual categorisation values. This fn does the softmax to squish it to probabilities and then does the cross entropy. It doesn't matter here but cross-entropy sums/squishes matrices, for example a size [2,5] tensor will become a size [2,1] tensor.
  
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(),      
      optimizer='Adam', learning_rate=0.01) # We use the Adam optimiser with a learning rate of 0.01 to minise the loss. 
  
  return (      
      {'class': tf.argmax(logits, 1), 
       'prob': tf.nn.softmax(logits)},      
      loss, train_op) # the outputs are the highest probability output of the logit vector (ie the calculated categorisation), the softmax function is a tensor the same size as logits and is the softmax fn applied to the logit vector. we also know the loss and 

# useful links:
# https://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with
# https://stackoverflow.com/questions/33681517/tensorflow-one-hot-encoder
# https://pandas.pydata.org/pandas-docs/stable/dsintro.html
# https://spark.apache.org/docs/latest/ml-features.html#word2vec
# https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected
# https://www.tensorflow.org/api_docs/python/tf/argmax
