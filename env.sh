#!/bin/bash
path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$path/python:$path/build/lib:$PYTHONPATH
