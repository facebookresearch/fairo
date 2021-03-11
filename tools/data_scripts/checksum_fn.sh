#!/bin/sh

# saves sha1sum for CHECKSUM_FOLDER at SAVE_TO_PATH 
calculate_sha1sum() {
    CHECKSUM_FOLDER=$1
    SAVE_TO_PATH=$2
    echo $'\nCalculating checksum for' ${CHECKSUM_FOLDER}
    echo "Saving to ${SAVE_TO_PATH}"
    find $CHECKSUM_FOLDER -type f ! -name '*checksum*' ! -name '*MD*' \
       -not -path '*/\.*' | sort -z | shasum | tr -d '-' \
       | xargs > $SAVE_TO_PATH
    echo "Saved checksum " $(cat $SAVE_TO_PATH) $'\n'
}