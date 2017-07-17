#!/bin/bash

cmd=../../increase_picture.py

echo processing...
for f in *.jpg *.jpeg *.JPG *.JPEG *.png *.PNG
do
	if [ -f $f ]; then
		echo python $cmd $f
		python $cmd $f
	fi
done

echo done
