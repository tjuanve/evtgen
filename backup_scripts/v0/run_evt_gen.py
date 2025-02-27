#!/mnt/ceph1-npx/user/tvaneede/software/py_venvs/event_generator_py3-v4.3.0/bin/python

import sys, os
print("Python Version:", sys.version)
print("Python Version (tuple):", sys.version_info)  # More structured output
print("Python Executable Path:", sys.executable)
print("Python Module Search Paths:", sys.path)
python_path = os.getenv("PYTHONPATH", "Not Set")
print("PYTHONPATH:", python_path)

# icecube imports
from icecube import dataio, icetray, dataclasses
from icecube import phys_services, photonics_service, millipede, VHESelfVeto
from icecube.photonics_service import I3PhotoSplineService
from icecube.dataclasses import I3Double, I3Particle, I3Direction, I3Position, I3VectorI3Particle, I3Constants, I3VectorOMKey
from icecube.dataclasses import I3RecoPulse, I3RecoPulseSeriesMap, I3RecoPulseSeriesMapMask, I3TimeWindow, I3TimeWindowSeriesMap
from icecube.icetray import I3Units, I3Frame, I3ConditionalModule, traysegment
from I3Tray import I3Tray
from icecube.millipede import MonopodFit, MuMillipedeFit, TaupedeFit, HighEnergyExclusions, MillipedeFitParams
from icecube.sim_services.label_events import MCLabeler, MuonLabels, CascadeLabels
from icecube.sim_services.label_events import ClassificationConverter

from egenerator.ic3.segments import ApplyEventGeneratorReconstruction

# python system imports
import sys, os, datetime
from glob import glob
from os.path import join
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pickle
from optparse import OptionParser

parser = OptionParser()
#parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
usage = """%prog [options]"""
parser.set_usage(usage)

parser.add_option("-i","--infiles",type=str,help="Input Files",dest="infiles",action='store',default='/data/ana/Diffuse/GlobalFit_Flavor/taupede/SnowStorm/RecowithBfr/Baseline/22086/0000000-0000999/Reco_NuTau_000990_out.i3.bz2')
parser.add_option("-o","--outfile",type=str,help="Output File",dest="outfile",action='store',default='/data/user/tvaneede/GlobalFit/EventGenerator/RecoEvtGen_NuTau_000990_out.i3.bz2')
# parser.add_option("-g","--gcdfile",type=str,help="GCD File",dest="GCDfile",action='store',default='/data/user/tvaneede/GlobalFit/EventGenerator/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz')
parser.add_option("--flavor", choices=('NuE', 'NuMu', 'NuTau','MuonGun','data') ,dest='flavor')
parser.add_option("--icemodel", choices=('Spice_3.2.1','Bfr'),default='Bfr', help="Specify which ice model to use",dest='icemodel')
parser.add_option("--year",type=str ,help="Year of operation.",default='IC86_2011',dest='year')
parser.add_option("--innerboundary", type=float, default=550., help="Inner detector boundary to determine contained energy depositions.",dest='innerboundary')
parser.add_option("--outerboundary", type=float, default=650., help="Outer detector boundary to determine contained energy depositions.",dest='outerboundary')

(opts,args) = parser.parse_args()
outfile = opts.outfile
# if not opts.GCDfile=='None':
#     gcdFile=opts.GCDfile
#     infiles=[gcdFile, opts.infiles]
# else:
#     print('No GCDFile')
#     infiles=opts.infiles

infiles=opts.infiles


# initialize timers
starttimes, stoptimes = {}, {}
timekeys = ['EventGenerator']
for timekey in timekeys:
    starttimes[timekey] = []
    stoptimes[timekey] = []


################################################################
########################## ICETRAY #############################
################################################################
starttime = datetime.datetime.now()
tray = I3Tray()

print("reading infiles", infiles)

tray.AddModule('I3Reader', 'reader', Filename=infiles)
# tray.AddModule('I3Reader', 'reader', FilenameList=infiles)

def timer(frame,tag,key):
    if tag == 'start':
        starttimes[key].append(datetime.datetime.now())
    elif tag == 'stop':
        stoptimes[key].append(datetime.datetime.now())


def filter_nullsplit(frame):
    if frame["I3EventHeader"].sub_event_stream=='NullSplit':
        return False
    else:
        eventid = frame['I3EventHeader'].event_id
        #print('Interaction tyoe is',frame['I3MCWeightDict']['InteractionType'])
        print("*******Currently processing frame %s*******" %eventid)


tray.Add(filter_nullsplit)
 
pulses = 'SplitInIcePulses'
photons_per_bin = 5
shower_spacing = 5

# Icemodel stuff
ice_model = opts.icemodel

if ice_model == 'Spice_3.2.1':
    base = os.path.expandvars('/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/cascade_single_spice_3.2.1_flat_z20_a5.%s.fits')
    base_eff = os.path.expandvars('/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/cascade_effectivedistance_spice_3.2.1_z20.%s.fits')
    tiltdir = os.path.expandvars('/cvmfs/icecube.opensciencegrid.org/users/vbasu/meta-projects/combo3/icetray/photonics-service/resources/tilt/')
    pxs = I3PhotoSplineService(base % "abs", base % "prob", effectivedistancetable=base_eff % "eff", timingSigma=0, tiltTableDir=tiltdir)
    #uncomment this following line if effectove distance is NOT to be used, by default, one should always use it for this script!
    #pxs = I3PhotoSplineService(base % "abs", base % "prob", timingSigma=0, tiltTableDir=tiltdir)
elif ice_model == 'Bfr':  
      base = os.path.expandvars('/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/cascade_single_spice_bfr-v2_flat_z20_a5.%s.fits')
      base_eff = os.path.expandvars('/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/cascade_effectivedistance_spice_bfr-v2_z20.%s.fits')
      tiltdir = os.path.expandvars('/cvmfs/icecube.opensciencegrid.org/users/vbasu/meta-projects/combo3/icetray/photonics-service/resources/tilt/')
      pxs = I3PhotoSplineService(base % "abs", base % "prob", effectivedistancetable=base_eff % "eff", timingSigma=0, tiltTableDir=tiltdir)
      #uncomment this following line if effectove distance is NOT to be used, by default, one should always use it for this script!
      #pxs = I3PhotoSplineService(base % "abs", base % "prob", timingSigma=0, tiltTableDir=tiltdir)
      
tray.Add('Delete', keys=['BrightDOMs', 'DeepCoreDOMs', 'SaturatedDOMs'])
excludedDOMs = tray.Add(HighEnergyExclusions,
    Pulses = pulses,
    BadDomsList = 'BadDomsList',
    CalibrationErrata = 'CalibrationErrata',
    ExcludeBrightDOMs = 'BrightDOMs',
    ExcludeDeepCore = False,
    ExcludeSaturatedDOMs = 'SaturatedDOMs',
    SaturationWindows = 'SaturationTimes') 

from parameters_evtgen import default_dnn_cascade_selection

parameters_evtgen = default_dnn_cascade_selection["EGen_2Cascade_Reco_config"]

################################################################
####################### HELPER CLASSES #########################
################################################################
# if opts.GCDfile=='None':
#     infile = dataio.I3File(infiles)
#     frame=infile.pop_frame(icetray.I3Frame.Geometry)
    
#     geometry = frame['I3Geometry'].omgeo
    
# else:
#     gcdfile = dataio.I3File(opts.GCDfile)
#     frame = gcdfile.pop_frame()
#     while 'I3Geometry' not in frame:
#         frame = gcdfile.pop_frame()
#     geometry = frame['I3Geometry'].omgeo

if opts.year == 'IC79_2010':
    strings = [2, 3, 4, 5, 6, 13, 21, 30, 40, 50, 59, 67, 74, 73, 72, 78, 77, 76, 75, 68, 60, 51, 41, 32, 23, 15, 8]
else:
    strings = [1, 2, 3, 4, 5, 6, 13, 21, 30, 40, 50, 59, 67, 74, 73, 72, 78, 77, 76, 75, 68, 60, 51, 41, 31, 22, 14, 7]

# outerbounds = {}
# cx, cy = [], []
# for string in strings:
#     omkey = icetray.OMKey(string, 1)
#     if geometry.has_key(omkey):
#         x, y = geometry[omkey].position.x, geometry[omkey].position.y
#         outerbounds[string] = (x, y)
#         cx.append(x)
#         cy.append(y)
# cx, cy = np.asarray(cx), np.asarray(cy)
# order = np.argsort(np.arctan2(cx, cy))
# outeredge_x = cx[order]
# outeredge_y = cy[order]

################################################################
    ############## Event Generator RECONSTRUCTION ################
################################################################
tray.Add(timer, tag='start', key='EventGenerator')

def add_taupede_seed(frame):
    seed_map = dataclasses.I3MapStringDouble()
    taupede1 = frame["HESETaupedeFit1"]
    taupede2 = frame["HESETaupedeFit2"]
    seed_map["x"] = taupede1.pos.x
    seed_map["y"] = taupede1.pos.y
    seed_map["z"] = taupede1.pos.z
    seed_map["time"] = taupede1.time
    seed_map["zenith"] = taupede1.dir.zenith
    seed_map['azimuth'] = taupede1.dir.azimuth
    seed_map['energy'] = taupede1.energy
    seed_map['cascade_00001_distance'] = taupede1.length
    seed_map['cascade_00001_energy'] = taupede2.energy
    frame["TaupedeSeed"] = seed_map

tray.Add(add_taupede_seed, "add_taupede_seed_for_egen")



def add_monopod_seeds(frame):
    seed_map = dataclasses.I3MapStringDouble()
    monopod = frame["HESEMonopodFit"]
    seed_map["x"] = monopod.pos.x
    seed_map["y"] = monopod.pos.y
    seed_map["z"] = monopod.pos.z
    seed_map["time"] = monopod.time
    seed_map["zenith"] = monopod.dir.zenith
    seed_map['azimuth'] = monopod.dir.azimuth
    seed_map['energy'] = monopod.energy/2
    seed_map['cascade_00001_distance'] = 50
    seed_map['cascade_00001_energy'] = monopod.energy/2
    frame[f"MonopodSeed_length50"] = seed_map

    seed_map100 = seed_map.copy()
    seed_map100['cascade_00001_distance'] = 100
    frame[f"MonopodSeed_length100"] = seed_map100

    seed_map200 = seed_map.copy()
    seed_map200['cascade_00001_distance'] = 200
    frame[f"MonopodSeed_length200"] = seed_map200

tray.Add(add_monopod_seeds, f"add_monopod_seeds_for_egen")


tray.AddSegment(
    ApplyEventGeneratorReconstruction, 'ApplyEventGeneratorReconstruction',
    pulse_key='SplitInIceDSTPulses',
    dom_and_tw_exclusions=['BadDomsList', 'CalibrationErrata', 'SaturationWindows'],
    partial_exclusion=True,
    exclude_bright_doms=True,
    model_base_dir="/data/ana/PointSource/DNNCascade/utils/exported_models/version-0.0/egenerator/",
    model_names=['starting_multi_cascade_7param_noise_tw_BFRv1Spice321_low_mem_n002_01'],
    seed_keys=['TaupedeSeed', 'MonopodSeed_length50', 'MonopodSeed_length100', 'MonopodSeed_length200'],
    output_key='MyEgeneratorOutputFrameKey',
)

# EventGenerator_cascade_7param_noise_tw_BFRv1Spice321_01 starting_multi_cascade_7param_noise_tw_BFRv1Spice321_low_mem_n002_01

# from dnn_cascade_selection.utils.reconstruction import run_egen_2_cascade_reco


tray.Add(timer, tag='stop', key='EventGenerator')

################################################################
   ########### Write ###########
################################################################

deletekeys =['']

tray.Add('Delete', keys=deletekeys)                                                                   

tray.AddModule('I3Writer', 'writer',DropOrphanStreams=[icetray.I3Frame.DAQ],
            #    Streams=[icetray.I3Frame.Geometry, icetray.I3Frame.Calibration,
               Streams=[icetray.I3Frame.Calibration,
                        icetray.I3Frame.DetectorStatus, icetray.I3Frame.DAQ, icetray.I3Frame.Physics],
                #Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics,icetray.I3Frame.Simulation,
              #icetray.I3Frame.Stream('M')],
                   Filename=opts.outfile)

tray.AddModule('TrashCan', 'yeswecan')
tray.Execute()
tray.Finish()
duration = datetime.datetime.now() - starttime
print("\t\tFinished I3Tray..")
print("")
print("This took:",duration)
print("")
print("Timing information for each modules is as follows:")
print("")
for timekey in timekeys:

    if len(starttimes[timekey]) == 0:
        continue
    tstart, tstop = np.asarray(starttimes[timekey]), np.asarray(stoptimes[timekey])
    if len(tstart) != len(tstop):
        durations = tstop - tstart[:len(tstop)]
    else:
        durations = tstop - tstart

    print ("\t{} took {}".format(timekey,durations.sum()))
