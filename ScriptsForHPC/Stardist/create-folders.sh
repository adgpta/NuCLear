#!/bin/bash
for x in ./*.tif; do
	mkdir "${x%.*}" && mv "$x" "${x%.*}"
done
