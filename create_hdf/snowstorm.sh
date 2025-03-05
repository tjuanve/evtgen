#!/bin/bash

set -e  # Exit on error

# neha
/cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/metaprojects/icetray/v1.4.1/env-shell.sh /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/bin/python /data/user/tvaneede/GlobalFit/EventGenerator/create_hdf/FullPID_NNMFit_hdf.py $@