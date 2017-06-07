#!/bin/bash

DATA_DIR=./deep_image_analogy/datasets

CONTENT_DIR=${DATA_DIR}/images_content
STYLE_DIR=${DATA_DIR}/images_style

ls ${CONTENT_DIR}/*.jpg >${DATA_DIR}/content_list.txt
ls ${STYLE_DIR}/*.jpg >${DATA_DIR}/style_list.txt

echo "Done!"