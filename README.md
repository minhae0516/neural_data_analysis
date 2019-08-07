# neural_data_analysis
This code is for neural data analysis using regression methods. 
All data files are located in `data` folder.


There are four codes to perform neural data analysis
- `data_preprocessing.py`: this code transform pkl file to Pandas DataFrame csv file. 
- `encoding_v3.ipynb`: Estimate belief from neural signal
    - input: r_df.csv (neural signal from 300 neurons)  
    - output: nb_df.csv (estimated neural belief)
    - method: linear regression
    - WORKS VERY GOOD!
 
- `decoding.ipynb`: Find policy that returns action from neural belief
    - input: nb_df.csv (estimated neural belief - obtained from `encoding_v3.ipynb`)     
    - output: a_df.csv (estimated neural belief)
    - method: multinomial logistic regression
    - DOES NOT WORK! 
    - tried iris data (`from sklearn import datasets`) in order to verify my code - WORKS GOOD 
    - We have a problem with our current action data set

- `recoding`: to be updated
