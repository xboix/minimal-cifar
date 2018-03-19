#!/bin/bash
cd ../log/
ls | grep -v ID0_base | xargs -t  rm -fr
cd ..
