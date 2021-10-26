#!/bin/bash

for filename in *.txt
do
	filename=$(basename -- "$filename" .txt)
	touch $filename.ann
done

