Henry Nguyen
Natural Language Processing Ass1
hnguye87

FILES:
   main.py : Program reads in data from 1b_benchmark.train.tokens to train the n-gram models.
    Uses 1b_benchmark.dev.tokens to test different weights for linear interpolation smoothing.
    Functions to find probability of sentences using n-gram models. Also functions to find complexity of
    n-gram models. 1b_benchmark.test.tokens are used to test for complexity at the end of program.
   
   1b_benchmark.train.tokens: token file used to train language model
   1b_benchmark.dev.tokens : token file used to test for the ideal weight for smoothing
   1b_benchmark.test.tokens : token file used to test complexity of functions
   

HOW_TO_RUN:
    The program consists of only main.py and requires 1b_benchmark.dev.tokens,
    1b_benchmark.test.tokens, and 1b_benchmark.train.tokens to run. The program will calculate the
    complexity of unigram, bigram, and trigram models with all three token files and print them out.
    In comments there is also code that tests the functions that calculate the probability of a sentence.
    These sentences are lines from the 3 token files.
