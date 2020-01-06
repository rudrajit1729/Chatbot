import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()

import numpy as np
import tensorflow as tf
import tflearn
import pickle
import json
import random

with open("intents.json") as file:
	data=json.load(file)

#print (data)
#print(data)
#try to load the encoded data if not load then load dataset and encode then proceed
#if u wanna load dataset again del the prev saved pickle file

try:
	#x#if u wanna train data again i.e. dataset has been updated
	with open("data.pickle","rb") as f:
		words,labels,training,output=pickle.load(f)
	print("Collection done!!!")
except:
	words=[]
	labels=[]
	docs_x=[]
	docs_y=[]

	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds=nltk.word_tokenize(pattern)
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(intent["tag"])

		if intent["tag"] not in labels:
			labels.append(intent["tag"])

	words=[stemmer.stem(w.lower()) for w in words if w!="?"]
	words=sorted(list(set(words)))
	labels=sorted(labels)

	training=[]
	output=[]
	out_empty=[0 for _ in range(len(labels))]

	for x,doc in enumerate(docs_x):
		bag=[]
		wrds=[stemmer.stem(w.lower()) for w in doc]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)
		output_row=out_empty[:]
		output_row[labels.index(docs_y[x])]=1

		training.append(bag)
		output.append(output_row)

	training=np.array(training)
	output=np.array(output)

	with open("data.pickle","wb") as f:
		pickle.dump((words,labels,training,output),f)
	print("Made new sets")

#Reseting the underlying graph data
tf.reset_default_graph()

#making the model with 2 hidden layers of 8 neurons each
net=tflearn.input_data(shape=[None,len(training[0])])
net=tflearn.fully_connected(net,8)#8 neurons for hidden layer 1
net=tflearn.fully_connected(net,8)#8 neurons for hidden layer 2
net=tflearn.fully_connected(net,len(output[0]),activation="softmax")#output layer
#softmax gives probabability to each neuron
net=tflearn.regression(net)

model=tflearn.DNN(net)# DNN-type of neural network

#try to load the trained model if not then train and save it
#if u wanna recreate the model then delete the prev model
try:
	model.load("model.tflearn")
	print("Model loaded successfully!!!")
except:
	model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)#n_epoch=no of times model sees the data
	model.save("model.tflearn")
	print("Model created successfully!!!")

def bag_of_words(s,words):
	#converts sentence to bag of words(those words in words list only picked from sentence)
	bag=[0 for x in range (len(words))]

	s_words=nltk.word_tokenize(s)
	s_words=[stemmer.stem(word.lower()) for word in s_words]

	for se  in s_words:
		for i,w in enumerate (words):
			if w==se:
				bag[i]=1
 
	return np.array(bag)

def chat():
	print("I am ready to chat.")
	while True:
		inp=input("You: ")
		results = model.predict([bag_of_words(inp ,words)])[0]
		results_index=np.argmax(results) #gives us the index of the greatest value in our list
		tag =labels[results_index]

		if results[results_index]>0.7:
			for intent in data["intents"]:
				if intent["tag"]==tag:
					responses=intent['responses']

			print(random.choice(responses))

			if tag=="goodbye":
				break
		else:
			print("I don't quite understand.Try again or ask a different question.")
			

chat()














