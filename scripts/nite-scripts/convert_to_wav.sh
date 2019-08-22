#!/bin/bash

if [ ! -f "$1" ]; then
	echo "Input file doesn't exist!"
	exit 1
fi

if [ -z "$2" ]; then
	echo "Please specify the output file"
	exit 1
fi

ffmpeg -i "$1" -acodec pcm_s16le -ac 1 -ar 16000 "$2"

