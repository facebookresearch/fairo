#!/bin/sh

# saves sha1sum for CHECKSUM_FOLDER at SAVE_TO_PATH 
calculate_sha1sum() {
    CHECKSUM_FOLDER=$1
    SAVE_TO_PATH=$2
    echo $'\nCalculating checksum for' ${CHECKSUM_FOLDER}
    # -exec calculates shasum for each file
    # cut trims the output of exec to exclude file names and finally pipes all shasums to the final shasum
    find $CHECKSUM_FOLDER -type f ! -name '*checksum*' ! -name '*MD*' \
       -not -path '*/\.*' -exec shasum {} \; | cut -d" " -f1 | sort -d | shasum | tr -d '-' | xargs > $SAVE_TO_PATH
    echo "Saving to ${SAVE_TO_PATH}"
    echo "Saved checksum " $(cat $SAVE_TO_PATH) $'\n'
}