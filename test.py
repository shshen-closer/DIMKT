# -*- coding:utf-8 -*-
__author__ = 'shshen'

import os
import sys
import time
import numpy as np 
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from datetime import datetime 
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from utils import checkmate as cmm
from utils import data_helpers as dh
import json
# Parameters
# ==================================================

logger = dh.logger_fn("tflog", "logs/test-{0}.log".format(time.asctime()).replace(':', '_'))
file_name = sys.argv[1]

MODEL = file_name
while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input("The format of your input is illegal, it should be like(90175368), please re-input: ")
logger.info("The format of your input is legal, now loading to next step...")



MODEL_DIR =  'runs/' + MODEL + '/checkpoints/'
BEST_MODEL_DIR =  'runs/' + MODEL + '/bestcheckpoints/'
SAVE_DIR = 'results/' + MODEL

# Data Parameters
tf.compat.v1.flags.DEFINE_string("checkpoint_dir", MODEL_DIR, "Checkpoint directory from training run")
tf.compat.v1.flags.DEFINE_string("best_checkpoint_dir", BEST_MODEL_DIR, "Best checkpoint directory from training run")

# Model Hyperparameters
tf.compat.v1.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.compat.v1.flags.DEFINE_float("keep_prob", 0.2, "Keep probability for dropout")
tf.compat.v1.flags.DEFINE_integer("hidden_size", 256, "The number of hidden nodes (Integer)")
tf.compat.v1.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.compat.v1.flags.DEFINE_integer("batch_size", 64 , "Batch size for training.")
tf.compat.v1.flags.DEFINE_integer("seq_len", 100, "Number of epochs to train for.")

# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.compat.v1.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.compat.v1.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100


def test():

    # Load data
    logger.info("Loading data...")

    logger.info("Training data processing...")

    test_students = np.load("data/test.npy", allow_pickle=True)
    max_num_steps = 100
    max_num_skills = 265

    BEST_OR_LATEST = 'B'

    while not (BEST_OR_LATEST.isalpha() and BEST_OR_LATEST.upper() in ['B', 'L']):
        BEST_OR_LATEST = input("he format of your input is illegal, please re-input: ")
    if BEST_OR_LATEST == 'B':
        logger.info("Loading best model...")
        checkpoint_file = cmm.get_best_checkpoint(FLAGS.best_checkpoint_dir, select_maximum_value=True)
    if BEST_OR_LATEST == 'L':
        logger.info("latest")
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.compat.v1.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_p = graph.get_operation_by_name("input_p").outputs[0]
            target_p = graph.get_operation_by_name("target_p").outputs[0]
            input_sd = graph.get_operation_by_name("input_sd").outputs[0]
            input_d = graph.get_operation_by_name("input_d").outputs[0]
            input_kc = graph.get_operation_by_name("input_kc").outputs[0]
            x_answer = graph.get_operation_by_name("x_answer").outputs[0]
            target_sd = graph.get_operation_by_name("target_sd").outputs[0]
            target_d = graph.get_operation_by_name("target_d").outputs[0]
            target_kc = graph.get_operation_by_name("target_kc").outputs[0]
            target_index = graph.get_operation_by_name("target_index").outputs[0]
            target_correctness = graph.get_operation_by_name("target_correctness").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]
            pred = graph.get_operation_by_name("pred").outputs[0]
            
            data_size = len(test_students)
            index = 0
            actual_labels = []
            pred_labels = []
            sdiffs = []
            kcs = []
            diffs = []
            pbs = []
            leng = []
            while(index+FLAGS.batch_size < data_size):
                input_p_b = np.zeros((FLAGS.batch_size, max_num_steps))
                target_p_b = np.zeros((FLAGS.batch_size, max_num_steps))
                input_sd_b = np.zeros((FLAGS.batch_size, max_num_steps))
                input_d_b = np.zeros((FLAGS.batch_size, max_num_steps))
                input_kc_b = np.zeros((FLAGS.batch_size, max_num_steps,max_num_skills))
                target_p_b = np.zeros((FLAGS.batch_size, max_num_steps))

                x_answer_b = np.zeros((FLAGS.batch_size, max_num_steps))
                target_sd_b = np.zeros((FLAGS.batch_size, max_num_steps))
                target_d_b = np.zeros((FLAGS.batch_size, max_num_steps))
                target_kc_b =  np.zeros((FLAGS.batch_size, max_num_steps,max_num_skills))
                target_correctness_b = []
                target_index_b = []
                
                for i in range(FLAGS.batch_size):
                    student = test_students[index+i]
                    ppp = student[0]
                    problem_ids = student[1]
                    correctness = student[2]
                    problem_kcs = student[3]
                    len_seq = student[4]
                    i_s =  student[5]

                    for j in range(len_seq-1):
                        input_p_b[i,j] = ppp[j]
                        input_sd_b[i,j] = i_s[j]
                        input_d_b[i,j] = problem_ids[j]
                        input_kc_b[i, j, int(problem_kcs[j])] = 1
                        x_answer_b[i,j] = correctness[j]
                        target_p_b[i,j] = ppp[j + 1]
                        target_d_b[i,j] = problem_ids[j + 1]
                        target_sd_b[i,j] = i_s[j + 1]
                        target_kc_b[i, j, int(problem_kcs[j+1])] =  1
                        target_index_b.append(i*max_num_steps+j)
                        target_correctness_b.append(int(correctness[j+1]))
                        actual_labels.append(int(correctness[j+1]))

                index += FLAGS.batch_size

                feed_dict = {
                    input_p: input_p_b,
                    target_p: target_p_b,
                    input_sd: input_sd_b,
                    input_d: input_d_b,
                    input_kc:input_kc_b,
                    x_answer: x_answer_b,
                    target_sd: target_sd_b,
                    target_d: target_d_b,
                    target_kc: target_kc_b,
                    target_index: target_index_b,
                    target_correctness: target_correctness_b,
                    dropout_keep_prob: 0.0,
                    is_training: False
                }
                


                pred_b = sess.run(pred, feed_dict)
                
                pred_labels.extend(pred_b.tolist())


            rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
            auc = metrics.roc_auc_score(actual_labels, pred_labels)

            pred_score = np.greater_equal(pred_labels,0.5) 
            pred_score = pred_score.astype(int)
            pred_score = np.equal(actual_labels, pred_score)
            acc = np.mean(pred_score.astype(int))
            print("epochs {0}: rmse {1:g}  auc {2:g}  acc {3:g}".format(0,rmse, auc, acc))
            logger.info("epochs {0}: rmse {1:g}  auc {2:g}   acc {3:g}".format(0,rmse, auc, acc))                     




if __name__ == '__main__':
    test()
