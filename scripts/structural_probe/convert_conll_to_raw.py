#!/usr/bin/env python3

"""
Embarassingly simple (should I have written it in bash?) script
for turning conll-formatted files to sentence-per-line
whitespace-tokenized files.

Takes the filepath at sys.argv[1]; writes to stdout
"""

import sys
import argparse

argp = argparse.ArgumentParser()
argp.add_argument('--input_conll_filepath')
argp.add_argument('--output_path')

args = argp.parse_args()

buf = []

output_file = open(args.output_path, 'w')
for line in open(args.input_conll_filepath):
  if line.startswith('#'):
    continue
  if not line.strip():
    output_file.write(' '.join(buf) + '\n')
    buf = []
  else:
    buf.append(line.split('\t')[1])
if buf:
    output_file.write(' '.join(buf) + '\n')
