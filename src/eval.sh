#!/bin/bash

for i in {10..1000..10}
do 
	echo "model_ep$i"
	python evaluate.py $1 "model_ep$i"
done  
