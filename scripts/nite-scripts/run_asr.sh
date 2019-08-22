#!/bin/bash

usage="Usage: $(basename "$0") [AUDIO_FILE]
This script can be used to obtain ASR and segmenter outputs from a recording.

Parameters:
AUDIO_FILE	a compatible audio file with sample rate 16000
"

if [ -z "$LD_LIBRARY_PATH" ]; then
	echo "[WARNING] Don't forget to export LD_LIBRARY_PATH=(...pv-platform.../lib64) !";
fi

if [ ! -f "$1" ]; then
	echo "[Error] audio file does not exist"; 
	echo "$usage";
	exit 1;
fi

AUDIO_FILE="$1"

DIRECTORY=$(dirname "$1")
PREFIX=$(basename "$1" ".wav")

# run ASR
ebclient -r -f en-EU-lecture_KIT-hybrid -i en-EU -t unseg-text "$AUDIO_FILE" -w "$DIRECTORY/$PREFIX.ctm"

# run ASR+segmenter
ebclient -r -f en-EU-lecture_KIT-hybrid -i en-EU -t text "$AUDIO_FILE" > "$DIRECTORY/$PREFIX.txt"

echo "All done!"

