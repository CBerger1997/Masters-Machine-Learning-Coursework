# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:35:51 2020

@author: callum
"""

import random
import numpy as np
import math

#loads the classification dataset and splits the dataset into its corresponding features and output
def load_dataset(file, batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6, output_list):
    my_file = open(file, "r")
    content_list = my_file.readlines()
    content_list_split = []
    my_file.close()

    for line in content_list:
        content_list_split.append(line.strip().split("\n"))
                
    for data in content_list_split:
        split_data = data[0].split(",")
        batch_x1.append(split_data[0])
        batch_x2.append(split_data[1])
        batch_x3.append(split_data[2])
        batch_x4.append(split_data[3])
        batch_x5.append(split_data[4])
        batch_x6.append(split_data[5])
        output_list.append(split_data[6])
        
    quantify_dataset(batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6, output_list)
            
# quantifies the features and outputs

def quantify_dataset(batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6, output_list):
   
    for count, data in enumerate(batch_x1):
        batch_x1[count] = quantify_buying(data)
    for count, data in enumerate(batch_x2):
        batch_x2[count] = quantify_maint(data)
    for count, data in enumerate(batch_x3):
        batch_x3[count] = quantify_doors(data)
    for count, data in enumerate(batch_x4):
        batch_x4[count] = quantify_persons(data)
    for count, data in enumerate(batch_x5):
        batch_x5[count] = quantify_lug_boot(data)
    for count, data in enumerate(batch_x6):
        batch_x6[count] = quantify_safety(data)
    for count, data in enumerate(output_list):
        output_list[count] = quantify_output(data)

def quantify_buying(value):
    quantified_vals = {"low" : 0, "med" : 1, "high" : 2, "vhigh" : 3}
    return quantified_vals[value]

def quantify_maint(value):
    quantified_vals = {"low" : 0, "med" : 1, "high" : 2, "vhigh" : 3}
    return quantified_vals[value]

def quantify_doors(value):
    quantified_vals = {"2" : 0, "3" : 1, "4" : 2, "5more" : 3}
    return quantified_vals[value]

def quantify_persons(value):
    quantified_vals = {"2" : 0, "4" : 1, "more" : 2}
    return quantified_vals[value]

def quantify_lug_boot(value):
    quantified_vals = {"small" : 0, "med" : 1, "big" : 2}
    return quantified_vals[value]

def quantify_safety(value):
    quantified_vals = {"low" : 0, "med" : 1, "high" : 2}
    return quantified_vals[value]

def quantify_output(value):
    quantified_vals = {"unacc" : 0, "acc" : 1, "good" : 2, "vgood" : 3}
    return quantified_vals[value]

# Stacks in the inputs into a 2D list
def column_stack_inputs(batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6):
    dataset = []
    
    for i in range(len(batch_x1)):
        dataset.append([batch_x1[i], batch_x2[i], batch_x3[i], batch_x4[i], batch_x5[i], batch_x6[i]])
        
    return dataset

# Jumbles the data up
def jumble_dataset(dataset_x, dataset_y):
    
    new_dataset_x = [None] * len(dataset_x)
    new_dataset_y = [None] * len(dataset_y)
    
    for i in range(len(dataset_x)):
        insertion_index = random.randint(0, len(dataset_x) - 1)
        is_not_inserted = True
        
        while is_not_inserted:
            if insertion_index >= len(dataset_x):
                insertion_index = 0
            
            if new_dataset_x[insertion_index] is None:
                new_dataset_x[insertion_index] = dataset_x[i]
                new_dataset_y[insertion_index] = dataset_y[i]
                is_not_inserted = False
            
            else:
                insertion_index = insertion_index + 1
    
    return [new_dataset_x, new_dataset_y]

# Column stacks the inputs and splits the data into testing and training datasets
def split_testing_training(testing_proportion, batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6, output_y, multi_output_y):
    
    stacked_x = column_stack_inputs(batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6)
    
    unacc_set_x = []
    acc_set_x = []
    good_set_x = []
    vgood_set_x = []
    
    unacc_set_y = []
    acc_set_y = []
    good_set_y = []
    vgood_set_y = []
    
    # Sorts each instance of data into lists depending on the associated classifications
    for i in range(len(output_y)):
        
        if output_y[i] == quantify_output("unacc"):
            unacc_set_x.append(stacked_x[i])
            unacc_set_y.append(multi_output_y[i])
            
        elif output_y[i] == quantify_output("acc"):
            acc_set_x.append(stacked_x[i])
            acc_set_y.append(multi_output_y[i])
        
        elif output_y[i] == quantify_output("good"):
            good_set_x.append(stacked_x[i])
            good_set_y.append(multi_output_y[i])
        
        elif output_y[i] == quantify_output("vgood"):
            vgood_set_x.append(stacked_x[i])
            vgood_set_y.append(multi_output_y[i])
    
    # Sorts unacc classified data into testing and training sets
    split_i = int(len(unacc_set_x) * testing_proportion)
    testing_set_x = unacc_set_x[:split_i]
    training_set_x = unacc_set_x[split_i:]
    testing_set_y = unacc_set_y[:split_i]
    training_set_y = unacc_set_y[split_i:]
    
    # Sorts acc classified data into testing and training sets
    split_i = int(len(acc_set_x) * testing_proportion)
    testing_set_x = testing_set_x + acc_set_x[:split_i]
    training_set_x = training_set_x + acc_set_x[split_i:]
    testing_set_y = testing_set_y + acc_set_y[:split_i]
    training_set_y = training_set_y + acc_set_y[split_i:]
    
    # Sorts good classified data into testing and training sets
    split_i = int(len(good_set_x) * testing_proportion)
    testing_set_x = testing_set_x + good_set_x[:split_i]
    training_set_x = training_set_x + good_set_x[split_i:]
    testing_set_y = testing_set_y + good_set_y[:split_i]
    training_set_y = training_set_y + good_set_y[split_i:]
    
    # Sorts vgood classified data into testing and training sets
    split_i = int(len(vgood_set_x) * testing_proportion)
    testing_set_x = testing_set_x + vgood_set_x[:split_i]
    training_set_x = training_set_x + vgood_set_x[split_i:]
    testing_set_y = testing_set_y + vgood_set_y[:split_i]
    training_set_y = training_set_y + vgood_set_y[split_i:]
    
    jumbled_testing = jumble_dataset(testing_set_x, testing_set_y)
    jumbled_training = jumble_dataset(training_set_x, training_set_y)
    
    return [jumbled_testing[0], jumbled_testing[1], jumbled_training[0], jumbled_training[1]]

def split_dataset_to_folds(folds, batch_x, batch_y):
     
    jumbled_dataset = jumble_dataset(batch_x, batch_y)
    
    dataset_chunks_x = list()
    dataset_chunks_y = list()

    dataset_x = jumbled_dataset[0]
    dataset_y = jumbled_dataset[1]

    jump = math.floor(len(batch_x) / folds)

    for i in range(folds):
        ind = i * jump
        dataset_chunks_x.append(dataset_x[ind:ind + jump])
        dataset_chunks_y.append(dataset_y[ind:ind + jump])
        
    return [dataset_chunks_x, dataset_chunks_y]