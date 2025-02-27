#!/bin/bash

/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/metaprojects/icetray/v1.12.0/env-shell.sh

source /data/user/tvaneede/software/py_venvs/event_generator_py3-v4.3.0/bin/activate

set -e  # Exit on error

/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/metaprojects/icetray/v1.12.0/env-shell.sh /mnt/ceph1-npx/user/tvaneede/software/py_venvs/event_generator_py3-v4.3.0/bin/python /data/user/tvaneede/GlobalFit/EventGenerator/run_evt_gen.py $@

# echo "which python?"

# which python
# which /mnt/ceph1-npx/user/tvaneede/software/py_venvs/event_generator_py3-v4.3.0/bin/python

# # Restore PYTHONPATH if it was cleared
# export PYTHONPATH=/mnt/ceph1-npx/user/tvaneede/software/py_venvs/event_generator_py3-v4.3.0/lib/python3.11/site-packages:/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/metaprojects/icetray/v1.12.0/lib:/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/metaprojects/icetray/v1.12.0/cernroot/lib/root:/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/lib/python3.11/site-packages

# /mnt/ceph1-npx/user/tvaneede/software/py_venvs/event_generator_py3-v4.3.0/bin/python /data/user/tvaneede/GlobalFit/EventGenerator/run_evt_gen.py $@