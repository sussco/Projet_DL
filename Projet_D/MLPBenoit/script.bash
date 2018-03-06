#!/bin/bash
python truc.py 2> truc.res
grep -o 'accuracy='[0:9]'.'[0-9]* truc.res > resulttrie.res
grep -o [0:1]'.'[0-9]* resulttrie.res > resulttrie2.res
python3 script_python/script.py
