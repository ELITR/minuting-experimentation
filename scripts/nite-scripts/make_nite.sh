#!/bin/bash

usage="Usage: $(basename "$0") [IN_TXT] [IN_CTM] [OUT_XML]
This script can be used to postprocess the output of the ASR and segmenter to create a NITE XML that can be used by annotators.

Parameters:
IN_TXT		path to segmenter output
IN_CTM		path to ASR output
OUT_XML		where to save the XML file

"

IN_TXT="$1"
IN_CTM="$2"
OUT_XML="$3"

SCRIPT_DIR=$(dirname "$0")


while getopts 'srd:f:' c; do
	case $c in
		h) echo "$usage"; exit 0 ;;
	esac
done

if [ ! -f "$IN_TXT" ]; then
	echo "[ERROR] IN_TXT path does not exist"
	echo "$usage"
	exit 1
fi

if [ ! -f "$IN_CTM" ]; then
	echo "[ERROR] IN_CTM path does not exist"
	echo "$usage"
	exit 1
fi

if [ -z "$OUT_XML" ]; then
	echo "[ERROR] OUT path not provided"
	echo "$usage"
	exit 1
fi

# get rid of intermediate ASR outputs and junk
cat "$IN_CTM" | python3 $SCRIPT_DIR/hypo_resolve.py > "$IN_CTM.clean" || exit 1

# get rid of intermediate segmenter outputs and junk
cat "$IN_TXT" | python3 $SCRIPT_DIR/filter_lines.py > "$IN_TXT.tokens" || exit 1

# merge ASR and segmenter outputs and create the XML file
# if this fails, it's possible that segmenter data still contains junk
python3 $SCRIPT_DIR/merge_xml.py "$IN_TXT.tokens" "$IN_CTM.clean" "$OUT_XML" || exit 1

# remove temp files
rm "$IN_CTM.clean"
rm "$IN_TXT.tokens"

echo "Done!"
