#!/bin/bash

nvcc -arch=sm_61 -x cu -std=c++17 --expt-extended-lambda lambdaKernel.cpp -o lambdaKernel.exe
