# -*- coding:utf-8 -*-
__author__ = 'shshen'

import numpy as np
import tensorflow as tf

def weight_variable(shape,  name=None, training = None):
    initial = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()(shape) , trainable=True, name=name)  
    return initial
def bias_variable(shape,  name=None, training = None):
    initial = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()(shape) , trainable=True, name=name) 
    return initial

class DIM(object):

    def __init__(self, batch_size, num_steps, num_skills, hidden_size):
        
        self.batch_size = batch_size = batch_size
        self.hidden_size  = hidden_size
        self.num_steps = num_steps
        self.num_skills =  num_skills

        self.input_p = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps], name="input_p")
        self.target_p = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps], name="target_p")
        self.input_sd = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps], name="input_sd")
        self.input_d = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps], name="input_d")
        self.input_kc = tf.compat.v1.placeholder(tf.float32, [batch_size, num_steps, num_skills], name="input_kc")
        self.x_answer = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="x_answer")
        self.target_sd = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="target_sd")
        self.target_d = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="target_d")
        self.target_kc = tf.compat.v1.placeholder(tf.float32, [batch_size, num_steps, num_skills], name="target_kc")
        self.target_index = tf.compat.v1.placeholder(tf.int32, [None], name="target_index")
        self.target_correctness = tf.compat.v1.placeholder(tf.float32, [None], name="target_correctness")


        
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
        
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.initializer=tf.compat.v1.keras.initializers.glorot_uniform()
        
        
        # problem
        self.p_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([53091, hidden_size]),dtype=tf.float32, trainable=True, name = 'p_w')
        zero_p = tf.zeros((1, hidden_size))
        all_p = tf.concat([zero_p, self.p_w ], axis = 0)
        p_embedding =  tf.nn.embedding_lookup(all_p, self.input_p)
        target_p =  tf.nn.embedding_lookup(all_p, self.target_p)


        self.sd_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([1010, hidden_size]),dtype=tf.float32, trainable=True, name = 'sd_w')
        zero_sd = tf.zeros((1, hidden_size))
        all_sd = tf.concat([zero_sd, self.sd_w ], axis = 0)
        sd_embedding =  tf.nn.embedding_lookup(all_sd, self.input_sd)
        target_sd =  tf.nn.embedding_lookup(all_sd, self.target_sd)


        skill_embedding =  tf.compat.v1.layers.dense(self.input_kc, units = hidden_size)
        target_skill = tf.compat.v1.layers.dense(self.target_kc, units = hidden_size)

        self.difficulty_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([1010, hidden_size]),dtype=tf.float32, trainable=True, name = 'difficulty_w')
        zero_difficulty = tf.zeros((1, hidden_size))
        all_difficulty = tf.concat([zero_difficulty, self.difficulty_w ], axis = 0)
        difficulty_embedding =  tf.nn.embedding_lookup(all_difficulty, self.input_d)
        target_difficulty =  tf.nn.embedding_lookup(all_difficulty, self.target_d)

        self.answer_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([2, hidden_size]),dtype=tf.float32, trainable=True, name = 'answer_w')
        x_answer =  tf.nn.embedding_lookup(self.answer_w, self.x_answer)




        input_data = tf.concat([p_embedding, skill_embedding, sd_embedding, difficulty_embedding], axis = -1)
        input_data =  tf.compat.v1.layers.dense(input_data, units = hidden_size)

        input_embedding = tf.add(input_data,0, name = 'input_embedding')


        target_data = tf.concat([target_p, target_skill, target_sd, target_difficulty], axis = -1)
        target_data = tf.compat.v1.layers.dense(target_data, units = hidden_size) 



        self.knowledge_w = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([1, hidden_size]),dtype=tf.float32, trainable=True, name = 'knowledge_matrix')


        kkk = tf.tile(self.knowledge_w, [batch_size,  1])



        shape = sd_embedding.get_shape().as_list()
        padd = tf.zeros((shape[0], 1, shape[2]))
        sd_embedding = tf.concat([padd, sd_embedding], axis = 1)
        slice_sd_embedding = tf.split(sd_embedding, self.num_steps+1, 1)

       
        
        shape = x_answer.get_shape().as_list()
        padd = tf.zeros((shape[0], 1, shape[2]))
        x_answer = tf.concat([padd, x_answer], axis = 1)
        slice_x_answer = tf.split(x_answer, self.num_steps+1, 1)

        shape = input_data.get_shape().as_list()
        padd = tf.zeros((shape[0], 1, shape[2]))
        input_data = tf.concat([padd, input_data], axis = 1)
        slice_input_data = tf.split(input_data, self.num_steps + 1, 1)


        input_diff = tf.concat([padd, difficulty_embedding], axis = 1)
        slice_input_diff = tf.split(input_diff, self.num_steps + 1, 1)



        h = list()



        w_l = weight_variable([shape[2], 1*hidden_size], name = 'w_l', training = self.is_training )
        b_l = bias_variable([shape[2]],  name='b_l', training = self.is_training )

        w_c = weight_variable([shape[2], 1*hidden_size], name = 'w_c', training = self.is_training )
        b_c = bias_variable([shape[2]],  name='b_c', training = self.is_training )


        w_o = weight_variable([shape[2], 2*hidden_size], name = 'w_o', training = self.is_training )
        b_o = bias_variable([shape[2]],  name='b_o', training = self.is_training )

        

        w_q = weight_variable([shape[2], 2*hidden_size], name = 'w_q', training = self.is_training )
        b_q = bias_variable([shape[2]],  name='b_q', training = self.is_training )

        w_b = weight_variable([shape[2], 4*hidden_size], name = 'w_b', training = self.is_training )
        b_b = bias_variable([shape[2]],  name='b_b', training = self.is_training )
        w_1 = weight_variable([shape[2], 4*hidden_size], name = 'w_1', training = self.is_training )
        b_1 = bias_variable([shape[2]],  name='b_1', training = self.is_training )


        

        for i in range(1,self.num_steps+1):

            sd = tf.squeeze(slice_sd_embedding[i], 1)

            aa = tf.squeeze(slice_x_answer[i], 1)

            dd = tf.squeeze(slice_input_diff[i], 1)

            q1 = tf.squeeze(slice_input_data[i], 1)  


            q = kkk - q1 

            input_gates = tf.sigmoid(tf.matmul(q,  tf.transpose(w_l, [1,0])+b_l), name='l_gate')
            c_title = tf.tanh(tf.matmul(q,  tf.transpose(w_c, [1,0])+b_c), name='c_gate')
            c_title = tf.nn.dropout(c_title, self.dropout_keep_prob)

            ccc = input_gates*c_title

            x = tf.concat([ccc, aa], axis = -1)
            xx = tf.sigmoid(tf.matmul(x,  tf.transpose(w_o, [1,0])+b_o), name='o_gate')
            xx_title = tf.tanh(tf.matmul(x,  tf.transpose(w_q, [1,0])+b_q), name='q_gate')

            xx =  xx*xx_title

            ins = tf.concat([kkk, aa,sd,dd], axis = -1)
            ooo = tf.sigmoid(tf.matmul(ins,  tf.transpose(w_1, [1,0])+b_1), name='1_gate')

            kkk = ooo*kkk +    (1-ooo) * xx 

            h_i = tf.expand_dims(kkk, axis = 1)
            h.append(h_i)

        output = tf.concat(h, axis = 1)
        logits = tf.reduce_sum(target_data*output, axis = -1, name="logits")
        logits = tf.reshape(logits, [-1])
        selected_logits = tf.gather(logits, self.target_index)
        
        #make prediction
        self.pred = tf.sigmoid(selected_logits, name="pred")

        # loss function
        losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_logits, labels=self.target_correctness), name="losses")

        l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.compat.v1.trainable_variables()],
                                 name="l2_losses") * 0.000001
        self.loss = tf.add(losses, l2_losses, name="loss")
        
        self.cost = self.loss