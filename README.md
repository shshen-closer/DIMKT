# DIMKT

Source code and data set for our paper (recently accepted in SIGIR2022): Assessing Student's Dynamic Knowledge State by Exploring the Question Difficulty Effect.

The code is the implementation of DIMKT model, and the data set is the public data set [ASSIST2012-2013](https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-withaffect).



## Dependencies:

- python >= 3.7
- tesorflow-gpu >= 2.0 
- numpy
- tqdm
- utils
- pandas
- sklearn


## Usage

First, download the data file: [2012-2013-data-with-predictions-4-final.csv](https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-withaffect), then put it in the folder 'data/' 

Then, run data_pre.py to preprocess the data set, and run data_save.py {sequence length} to divide the original data set into train set, validation set and test set. 

`python data_pre.py`


`python data_save.py 100`

Train the model:

`python train.py {fold}`

For example:

`python train.py 1`  or `python train.py 2`

Test the trained the model on the test set:

`python test.py {model_name}`


