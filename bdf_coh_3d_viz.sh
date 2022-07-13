#!/bin/bash
mkdir out
mkdir out/parts
mkdir out/parts_out
for i in {0..100}
do
  python bdf_coh_3d_viz.py
  cp out/parts_out/* out/parts
  rm out/parts_out/*
done
