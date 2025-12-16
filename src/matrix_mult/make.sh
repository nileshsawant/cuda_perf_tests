#!/bin/bash

nvcc -arch=sm_61 -x cu -std=c++17 --expt-extended-lambda matMult.cpp -o matMult.exe
