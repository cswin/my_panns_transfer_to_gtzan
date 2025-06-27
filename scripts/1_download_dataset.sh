#!/bin/bash
DATASET_DIR=${1:-"./datasets/audioset201906"}   # Default argument.

echo "------ Download metadata ------"
mkdir -p $DATASET_DIR"/metadata"

# Download video list csv.
wget -O $DATASET_DIR"/metadata/eval_segments.csv" "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"
wget -O $DATASET_DIR"/metadata/balanced_train_segments.csv" "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
wget -O $DATASET_DIR"/metadata/unbalanced_train_segments.csv" "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"

# Download class labels indices.
wget -O $DATASET_DIR"/metadata/class_labels_indices.csv" "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"

# Download quality of counts.
wget -O $DATASET_DIR"/metadata/qa_true_counts.csv" "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/qa/qa_true_counts.csv"

echo "Download metadata to $DATASET_DIR/metadata"

echo "------ Download wavs ------"
# Download evaluation wavs (limit to 10 files for testing)
python3 scripts/download_audioset.py \
    --csv_path=$DATASET_DIR"/metadata/eval_segments.csv" \
    --audios_dir=$DATASET_DIR"/audios/eval_segments" \
    # --max_files=10

# Download balanced train wavs (limit to 50 files for testing)
python3 scripts/download_audioset.py \
    --csv_path=$DATASET_DIR"/metadata/balanced_train_segments.csv" \
    --audios_dir=$DATASET_DIR"/audios/balanced_train_segments" \
    # --max_files=50

echo "Download completed! Check the logs for details."
echo "To download more files, remove the --max_files parameter or increase the number." 