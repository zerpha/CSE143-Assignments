Henry Nguyen
Natural Language Processing Ass3
hnguye87
   

HOW_TO_RUN:
	The python and google colab code should run without any special instructions.
	The second to last line in the code prints predicted accuracy on the dev set.
	To test on the test set for 3.3.1, switch ner.dev to ner.test.
	
	In 5.1.2, the output of the best model from early stopping is on the dev set 
	by default. To test on the test set, go to training_observer() inside 
	train() and switch ner.dev to ner.test to test and ner.dev.out
	to ner.test.out to try with test. 