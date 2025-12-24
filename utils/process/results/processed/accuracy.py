import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean, std, arange
from math import sqrt
from os import listdir
from pickle import dump

# figureType = 1 -> print loss
# figureType = otherwise -> print accuracy

# language = 1 -> print in portuguese-br
# language = otherwise -> print in english

def file_to_list(figureType=0,
                 epochs=40,
                 results_path='results/classification/random',
                 pattern="79871/79871"):

    
    file_names = {}
    result_files = {}
    file_lines = {}
    
    

    total_files = len(listdir(results_path))
        
        
    for index,name in enumerate(listdir(results_path)):
        file_names["filename"+str(index+1)] = results_path+name
            
    for i in range(1,total_files+1):
        result_files["result"+str(i)] = open(file_names['filename'+str(i)], 'r')


    for i in range(1,total_files+1):
        file_lines["Lines"+str(i)] = result_files['result'+str(i)].readlines()
    
        
    for i in range(1,total_files+1):
        result_files['result'+str(i)].close()

    accuracies = []
    ac = []
    

    for i in range(total_files):
        accuracies.append([])
        ac.append([])

    if figureType == 1:
        for i in range(1,total_files+1):
            for line in file_lines['Lines'+str(i)]:
                if ("/step" in line) and (pattern in line):
                    line = "".join(c for c in line if c.isprintable())
                    accuracies[i-1].append(float(line.split(':')[1][1:].split(' ')[0]))        
    else:
        for i in range(1,total_files+1):
            for line in file_lines['Lines'+str(i)]:
                if ("/step" in line) and (pattern in line):
                    line = "".join(c for c in line if c.isprintable())
                    accuracies[i-1].append(float(line.split(':')[2].split(' ')[1]))
       
    for i in range(total_files):
        ac[i] = accuracies[i][:]

    ac = [ele[:epochs] for ele in ac if ele != []]
    

    try:
        x1Mean = mean(ac,axis=0)
        x1Interval = std(ac,axis=0)*1.96/sqrt(80)
        return(x1Mean, x1Interval)    
    
    except Exception as e:
        
        print("error:", e)
        
        for i in ac:
            print(len(i))
    
   
if __name__ == "__main__":
    
    pattern="128/128"   
    datasets = [ "VeReMi", "WiSec" ]
    models = [ "m_fastest", "random" ]
    

    for model in models:
        
        for dataset in datasets:


            m,s = file_to_list(figureType=1,
                               epochs=40,
                               results_path='results/classification/raw/'+model+'/'+dataset+'/',
                               pattern=pattern)
            
            with open('results/classification/processed/'+model+'/'+dataset+'_mean_model','wb') as writer:
                dump(m, writer)

            with open('results/classification/processed/'+model+'/'+dataset+'_std_model','wb') as writer:
                dump(s, writer) 
    



