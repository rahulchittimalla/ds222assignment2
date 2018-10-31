from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.datasets import imdb
import argparse
import os
import random
from nltk.corpus import stopwords
import re
import time
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()

doc2id = {}
id2doc = {}
target2id = {}

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--train_file', type=str)
parser.add_argument('--dev_file', type=str)
parser.add_argument('--test_file', type=str)
parser.add_argument('--save_path', type=str)
args = parser.parse_args()
#file_name = '/scratch/ds222-2017/assignment-1/DBPedia.full/full_train.txt'


#vectorizer = TfidfVectorizer()
#response = vectorizer.fit_transform()

REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
        text = text.lower()
        text = REPLACE_BY_SPACE_RE.sub(' ', text)
        text = BAD_SYMBOLS_RE.sub('', text)
        text = ' '.join(word for word in text.split() if word not in STOPWORDS)
        return text

def read_file(filename, test_file=False):
    X = []
    y = []
    if test_file:
        with open(filename, 'r') as f:
            for line in f:
                line = line.split('\t')
                targets = line[0].split(',')
                doc = clean_text(line[1].split('"', 2)[1])
                t = []
                for target in targets:
                    target = target.strip()
                    if target not in target2id:
                        target2id[target] = len(target2id)
                    t.append(target2id[target])
                if len(t) < 1: print('Warning: no label')
                y.append(t)
                X.append(doc)
    else:
        with open(filename, 'r') as f:
            for line in f:
                line = line.split('\t')
                targets = line[0].split(',')
                doc = clean_text(line[1].split('"', 2)[1])
                for target in targets:
                    target = target.strip()
                    if target not in target2id:
                        target2id[target] = len(target2id)
                    y.append(target2id[target])
                    X.append(doc)
    return X, y
        
def get_tfidf(corpus):
        vectorizer =TfidfVectorizer()
        response = vectorizer.fit_transform(corpus)
        return vectorizer, response

def build_dict(X):
    for i, doc in enumerate(X):
        if doc not in doc2id:
            doc2id[doc] = i+1

    for k, v in doc2id.iteritems():
        id2doc[v] = k

def sigmoid(W, X):
    #logger.info('X.shape: {} W.shape: {}'.format(X.shape, W.shape))
    z = X.dot(W.T)
    return 1/(1 + np.exp(-z))

def loss(X, Y, W):
    H = sigmoid(W, X)

    #logger.info('y  .shape: {} h.shape: {}'.format(y.shape, h.shape))
    Z = Y * np.log(H) + (1-Y)*np.log(1-H)
    z = -np.mean(Z, 0)
    return z.mean(), z.std()

def gradient(X, Y, W):
    H = sigmoid(W, X)
    Z = X.T.dot(H-Y)
    Z /= len(Y)
    return Z.T

def predictions(X, Y, W):
    logger.info('Calculating sigmoid for predictions')
    H = sigmoid(W, X)
    #preds = np.zeros((len(y), 50))
    #preds[H > 0.5] = 1
    return np.argmax(H, 1)

def accuracy(X, Y, W):
    logger.info('Calculating predictions')
    preds = predictions(X, Y, W)
    acc = np.array(map(lambda x,y: x+1 in y, preds, Y))
    return acc.mean()

if __name__ == '__main__':
    
    logger.info('Loading the training file')
    X_train, y_train = read_file(args.train_file)
    y_train = np.array(y_train)
    logger.info('training set size: {}'.format(len(X_train)))
    logger.info('Loading the testing file')
    X_dev, y_dev = read_file(args.dev_file, test_file=True)
    y_dev = np.array(y_dev)
    logger.info('dev set size: {}'.format(len(X_dev)))
    logger.info('No of classes: {}'.format(len(target2id)))
    logger.info('building tf-idf matrix')
    start_time = time.time()
    vectorizer, response = get_tfidf(X_train+X_dev)
    tfidf = response.toarray()
    logger.info('time take to build tf-idf: {}'.format((time.time()-start_time)/60))
    logger.info('tfidf.shape: {}'.format(tfidf.shape))
    logger.info('building dictionary')
    build_dict(X_train+X_dev)
    logger.info('len(X_train): {}'.format(len(X_train)))
    logger.info('len(X_test): {}'.format(len(X_dev)))
    logger.info('len(X_train+X_test): {}'.format(len(X_train)+ len(X_dev)))
    logger.info('len(doc2id): {}'.format(len(doc2id)))
    logger.info('len(id2doc): {}'.format(len(id2doc)))
    logger.info('converting labels to one-hot encoding')
    dummy = np.zeros((y_train.size, y_train.max()+1))
    dummy[np.arange(y_train.size), y_train] = 1
    y_train = dummy

    n_batches = len(X_train) // args.batch_size
    n_classes = len(target2id)
    logger.info('No of batches: {}'.format(n_batches))
    print('Getting dev_indices')
    dev_indices =  map(lambda x: doc2id[x], X_dev[:10000])
    print('Getting X_dev_features')
    X_dev_features = np.array(tfidf[dev_indices])
    logger.info('X_dev_features.shape: {}'.format(X_dev_features.shape))
    
    cluster = tf.train.ClusterSpec({"ps":["localhost:65062"], "worker":["localhost:65063", "localhost:65064"]})
    ps = tf.train.Server(cluster, job_name='ps', task_index=0)
    worker0 = tf.train.Server(cluster, job_name='worker', task_index=0)
    worker1 = tf.train.Server(cluster, job_name='worker', task_index=1)

    seed = 2
    tf.set_random_seed(seed)
    m = tfidf.shape[1]
    n = n_classes
    
    worker0_X = X_train[:len(X_train)//2]
    worker1_X = X_train[len(X_train)//2:]
    worker0_y = y_train[:len(y_train)//2]
    worker1_y = y_train[len(y_train)//2:]
    
    # Declare the parameters on the parameter server ps
    with tf.device('/job:ps/task:0') as sess:
         W = tf.get_variable("W", shape=[m, n], initializer=tf.contrib.layers.xavier_initializer())
         b = tf.Variable(tf.zeros([n]))

    with tf.Session(worker0.target) as sess:

        x = tf.placeholder(tf.float32, [None, m])
        y_ = tf.placeholder(tf.float32, [None, n])

        y = tf.matmul(x, W) + b

        correct_prediction = tf.argmax(y, 1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
        opt = tf.train.GradientDescentOptimizer(args.lr)
        # This line is for synchronization

        opt = tf.trainSyncReplicasOptimizer(opt, replicas_to_aggregate=50, total_num_replicas=50)
        train_step = opt.minimize(cross_entropy)
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            epoch_time = time.time()

            for batch in range(n_batches-1):
                if batch != n_batches - 1:
                    start_index = batch * args.batch_size
                    end_index = start_index + args.batch_size
                    X_batch = worker0_X[start_index:end_index]
                    y_batch = np.array(worker0_y[start_index:end_index])
                else:
                    start_index = batch*args.batch_size
                    X_batch = worker0_X[start_index:]
                    y_batch = np.array(worker0_y[start_index:])

                indices = map(lambda x: doc2id[x], X_batch)
                X_features = np.array(tfidf[indices])

                _, loss = sess.run([train_step, cross_entropy], feed_dict={x:X_features, y_:y_batch})
                logger.info('epoch {}/{} batch {}/{} loss: {}'.format(epoch+1, args.epochs, batch+1, n_batches, loss))
                sess.join()
                if ((batch+1) % 100) == 0 :
                    preds = sess.run(correct_prediction, feed_dict={x:X_dev_features})
                    acc = np.array(map(lambda x,y: x in y, preds, y_dev[:10000]))
                    acc = acc.mean()
                    logger.info('Accuracy: {}'.format(acc))
                    with open(os.path.join(args.save_path, 'bsp.txt'), 'a+') as f:
                        line = '{} {} {} {}'.format(epoch+1, batch+1, loss, acc)
                        f.write(line + '\n')
    with tf.Session(worker1.target) as sess:

        x = tf.placeholder(tf.float32, [None, m])
        y_ = tf.placeholder(tf.float32, [None, n])

        y = tf.matmul(x, W) + b

        correct_prediction = tf.argmax(y, 1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
        opt = tf.train.GradientDescentOptimizer(args.lr)
        # This line is for synchronization

        opt = tf.trainSyncReplicasOptimizer(opt, replicas_to_aggregate=50, total_num_replicas=50)
        train_step = opt.minimize(cross_entropy)
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            epoch_time = time.time()

            for batch in range(n_batches-1):
                if batch != n_batches - 1:
                    start_index = batch * args.batch_size
                    end_index = start_index + args.batch_size
                    X_batch = worker1_X[start_index:end_index]
                    y_batch = np.array(worker1_y[start_index:end_index])
                else:
                    start_index = batch*args.batch_size
                    X_batch = worker1_X[start_index:]
                    y_batch = np.array(worker1_y[start_index:])

                indices = map(lambda x: doc2id[x], X_batch)
                X_features = np.array(tfidf[indices])

                _, loss = sess.run([train_step, cross_entropy], feed_dict={x:X_features, y_:y_batch})
                logger.info('epoch {}/{} batch {}/{} loss: {}'.format(epoch+1, args.epochs, batch+1, n_batches, loss))
                sess.join()
                if ((batch+1) % 100) == 0 :
                    preds = sess.run(correct_prediction, feed_dict={x:X_dev_features})
                    acc = np.array(map(lambda x,y: x in y, preds, y_dev[:10000]))
                    acc = acc.mean()
                    logger.info('Accuracy: {}'.format(acc))
                    with open(os.path.join(args.save_path, 'bsp.txt'), 'a+') as f:
                        line = '{} {} {} {}'.format(epoch+1, batch+1, loss, acc)
                        f.write(line + '\n')



           '''for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_time = time.time()
            
           

            logger.info('epoch {}/{} batch {}/{}, loss: {}'.format(epoch+1, args.epochs, batch+1, n_batches, batch_loss))
             

        epoch_loss /= n_batches
        logger.info('epoch: {}/{} loss: {}'.format(epoch+1, args.epochs, epoch_loss))
        logger.info('time per epoch: {}'.format((time.time()-epoch_time)/60))'''

        





