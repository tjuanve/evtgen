# IceCube imports
from icecube import dataio, icetray, dataclasses, simclasses
from icecube import millipede, MuonGun
from icecube.dataclasses import I3Double, I3Particle, I3Direction, I3Position, I3VectorI3Particle, I3Constants, I3String, I3UInt64
from icecube.dataclasses import I3RecoPulse, I3RecoPulseSeriesMap, I3RecoPulseSeriesMapMask
from icecube.icetray import I3Units, I3Frame, I3ConditionalModule, traysegment
from icecube import simclasses
from icecube.hdfwriter import I3HDFWriter
from I3Tray import I3Tray
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from glob import glob
import numpy as np
from icecube import dataio, hdfwriter, icetray
import sys, os, datetime
import pandas as pd
from os.path import join

sys.path.append('/data/user/tvaneede/GlobalFit/create_hdf_files')

from segments.TrueObservables import calculatetrueobservables
from segments.MCinfo_NL import mcinfo
from segments.MCPreProc import mcpreproc
from segments.MCInfoWrapper import MCInfoWrapper
from segments.PassingFraction import penetrating_depth, PassingFraction,add_primary
from segments.FinalEventClassification import checkfinaltopology, checkfinaltopology_evtgen
from segments.DeepCoreCharge import ChargeDepositions
import segments.meseUtils
import segments.MCWeightUtils

from I3Tray import *

parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--Dataset",type=str,help='Dataset Number of simulation set',dest="dataset")
parser.add_argument("--Topology",type=str,choices=('Cascades','DoubleCascades','Tracks'),help='Topology of the selected events',dest="topology")
parser.add_argument("--Folder",type=str,choices=('Baseline', 'OffBaseline', 'Perturbed','MuonGun'),dest="folder")
parser.add_argument('--do_60TeVcut',action="store_true",dest='do_60TeVcut',default=False)
parser.add_argument("--subfolder",type=str,dest="subfolder")
parser.add_argument("--Table",type=str,choices=('RecowithBfr','RecowithSpice321'),dest="table")

opts = parser.parse_args()


if opts.topology=='Cascades':
    topology=1
elif opts.topology=='DoubleCascades':
    topology=2
elif opts.topology=='Tracks':
    topology=3
else:
    topology=0
    print('Not a correct topology or spelling ;)')
dataset = opts.dataset
do_60TeVcut = opts.do_60TeVcut
print(do_60TeVcut)
filedir = "/data/user/tvaneede/GlobalFit/EventGenerator/reco_files/{0}/{1}/{2}/{3}".format(opts.table,opts.folder,opts.dataset,opts.subfolder)
gcdFile = "/data/user/nlad/Ternary_Classifier/Ternary_Classifier/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz"
files = [gcdFile]
files.extend(glob(filedir + '/*'))


#files.extend(glob(filedir + '{0}/*/*'.format(dataset)))
tray = I3Tray()
tray.Add("I3Reader", FileNameList=files)

################################################################
############## Energy cut and Final Event selections ###########
################################################################

'''
In Principle, this can also be done later, or Below60TeV flag can be added to the frame instead
This is being done here to make it more easier to create NNMFit dataframes
'''
if do_60TeVcut:
    tray.Add(lambda frame : 'RecoETot' in frame and frame['RecoETot'].value>=60000)


################################################################
####################### HELPER CLASSES #########################
################################################################

gcdfile = dataio.I3File(gcdFile)
frame = gcdfile.pop_frame()
while 'I3Geometry' not in frame:
    frame = gcdfile.pop_frame()
geometry = frame['I3Geometry'].omgeo


strings = [1, 2, 3, 4, 5, 6, 13, 21, 30, 40, 50, 59, 67, 74, 73, 72, 78, 77, 76, 75, 68, 60, 51, 41, 31, 22, 14, 7]

outerbounds = {}
cx, cy = [], []
for string in strings:
    omkey = icetray.OMKey(string, 1)
    if geometry.has_key(omkey):
        x, y = geometry[omkey].position.x, geometry[omkey].position.y
        outerbounds[string] = (x, y)
        cx.append(x)
        cy.append(y)
cx, cy = np.asarray(cx), np.asarray(cy)
order = np.argsort(np.arctan2(cx, cy))
outeredge_x = cx[order]
outeredge_y = cy[order]

def reclassify_double(frame):
    classification = frame['FinalTopology'].value
    if classification != 2:
        frame['FinalEventClass']= dataclasses.I3Double(classification)
    else:
        if frame['RecoL']<=20 and frame['RecoETot']>=3000000:
            frame['FinalEventClass']=dataclasses.I3Double(1)
        else:
            frame['FinalEventClass']= dataclasses.I3Double(classification)
    return True

tray.Add(reclassify_double)

tray.Add(checkfinaltopology_evtgen)

def reclassify_double_evtgen(frame):
    classification = frame['FinalTopology_evtgen'].value
    if classification != 2:
        frame['FinalEventClass_evtgen'] = dataclasses.I3Double(classification)
    else:
        if np.abs(frame['MyEgeneratorOutputFrameKey']['cascade_cascade_00001_distance'])<=20 and frame['RecoETot']>=3000000:
            frame['FinalEventClass_evtgen']=dataclasses.I3Double(1)
        else:
            frame['FinalEventClass_evtgen']= dataclasses.I3Double(classification)
    return True

tray.Add(reclassify_double_evtgen)

# #topology cut
# '''
# This is not any form of 'cut' just dividing each datasets in three topologies 
# which makes it convinient to use them in NNMFit framework

# topology = 1 (cascades)
# topology = 2 (double)
# topology = 3 (track)

# '''

# tray.Add(lambda frame : 'FinalEventClass' in frame and frame['FinalEventClass'].value == topology)


################################################################
    ############## Adding Passing fractions ##############
################################################################
def filter_nullsplit(frame):
    
    eventid = frame['I3EventHeader'].event_id
    runid = frame['I3EventHeader'].run_id
    print('Interaction type is',frame['I3MCWeightDict']['InteractionType'])
    print("*******Currently processing frame {0} in file {1}*******".format(eventid, runid))


tray.Add(filter_nullsplit)

tray.Add('Delete', keys=['TrueETot'])
def trueenergy(frame):
    neutrinos = [dataclasses.I3Particle.NuE, dataclasses.I3Particle.NuEBar,
                     dataclasses.I3Particle.NuMu, dataclasses.I3Particle.NuMuBar,
                     dataclasses.I3Particle.NuTau, dataclasses.I3Particle.NuTauBar]
    invisibletypes = [I3Particle.NuE, I3Particle.NuEBar,
                     I3Particle.NuMu, I3Particle.NuMuBar,
                     I3Particle.NuTau, I3Particle.NuTauBar,
                    I3Particle.MuPlus,I3Particle.MuMinus,
                    I3Particle.TauPlus,I3Particle.TauMinus,I3Particle.unknown]
    e=[]
    for p in frame['I3MCTree'].get_primaries():
        if p.type in neutrinos:
            for d in frame['I3MCTree'].get_daughters(p):
                
                for c in frame['I3MCTree'].get_daughters(d):
                    if c.shape != I3Particle.Dark and c.type not in invisibletypes:
                        e.append(c.energy)
    
    if not np.isfinite(sum(e)):
        etot = 1 # limit to 1 GeV
    else:
        etot = sum(e)

    frame.Put('TrueETot',I3Double(etot))
    return True
    
tray.Add(trueenergy)
tray.AddModule(add_primary)
tray.AddModule(penetrating_depth, gcd=gcdFile, depth_name_suffix='')

tray.Add(PassingFraction)


def recolbye(frame):
    length = frame['RecoL'].value
    energy = frame['RecoETot'].value
    lbye = length/energy
    frame['RecoLbyE'] = dataclasses.I3Double(length/energy)

tray.Add(recolbye)

def NNMFitVariables(frame):
    
    event = frame['FinalEventClass'].value 
     
    
    if event==1:
            frame['RecoParticle']=frame['HESEMonopodFit']
            frame['RecoEnergy']=dataclasses.I3Double(frame['RecoETot'].value)
            frame['RecoDirection']=dataclasses.I3Double(frame['RecoZenith'].value)
            frame['RecoLength']=dataclasses.I3Double(frame['RecoL'].value)
            frame['TrueLength'] = dataclasses.I3Double(frame['TrueL'].value)
            
            
    elif event==2:
            frame['RecoParticle']=frame['HESETaupedeFit']
            frame['RecoEnergy']=dataclasses.I3Double(frame['RecoETot'].value)
            frame['RecoDirection']=dataclasses.I3Double(frame['RecoZenith'].value)
            frame['RecoLength']=dataclasses.I3Double(frame['RecoL'].value)
            frame['TrueLength'] = dataclasses.I3Double(frame['TrueL'].value)
            
    elif event==3:
            frame['RecoParticle']=frame['SPEFit16']
            frame['RecoEnergy']=dataclasses.I3Double(frame['RecoETot'].value)
            frame['RecoDirection']=dataclasses.I3Double(frame['RecoZenith'].value)
            frame['TrueLength'] = dataclasses.I3Double(frame['TrueL'].value)
            


tray.AddModule(NNMFitVariables)



def vertex(frame):
    
    HESEMonopodFit = frame['HESEMonopodFit']
    HESETaupedeFit1 = frame['HESETaupedeFit1']
    HESETaupedeFit2 = frame['HESETaupedeFit2']
    
    frame['HESEMonopodFit_x']=dataclasses.I3Double(HESEMonopodFit.pos.x)
    frame['HESEMonopodFit_y']=dataclasses.I3Double(HESEMonopodFit.pos.y)
    frame['HESEMonopodFit_z']=dataclasses.I3Double(HESEMonopodFit.pos.z)

    frame['HESETaupedeFit1_x']=dataclasses.I3Double(HESETaupedeFit1.pos.x)
    frame['HESETaupedeFit1_y']=dataclasses.I3Double(HESETaupedeFit1.pos.y)
    frame['HESETaupedeFit1_z']=dataclasses.I3Double(HESETaupedeFit1.pos.z)

    frame['HESETaupedeFit2_x']=dataclasses.I3Double(HESETaupedeFit2.pos.x)
    frame['HESETaupedeFit2_y']=dataclasses.I3Double(HESETaupedeFit2.pos.y)
    frame['HESETaupedeFit2_z']=dataclasses.I3Double(HESETaupedeFit2.pos.z)
    
            
tray.AddModule(vertex)


#Following both functions
#for glashow correction and Tau Polarization are
# Taken from Aswathi
def glashow_correction(frame):
    
    from scipy.interpolate import interp1d
    nutype=int(frame["I3MCWeightDict"]["PrimaryNeutrinoType"])
    inter_type=int(frame['I3MCWeightDict']['InteractionType'])
    en = float(frame['I3MCWeightDict']['PrimaryNeutrinoEnergy'])
    if (abs(nutype)==12 and inter_type==3.0 and en>4e6):
        old_spline=pd.read_csv('/home/abalagopalv/diffuse/TauStudies/Glashow_old.csv',header=None)
        new_spline=pd.read_csv('/home/abalagopalv/diffuse/TauStudies/Glashow_new.csv',header=None)

        x = old_spline[0]
        y = old_spline[1]

        xn = new_spline[0]
        yn = new_spline[1]
        f1 = interp1d(x, y, kind='cubic')
        f2 = interp1d(xn, yn, kind='cubic')
        if en<9.9e6:

            num = f2(en/1e6)
            denom = f1(en/1e6)
            ratio = num/denom
            frame['TotalWeight'] = dataclasses.I3Double(frame['I3MCWeightDict']['TotalWeight']*ratio)
        elif en>=9.9e6:
            num = f2(9.89)
            denom = f1(9.89)
            ratio = num/denom
            frame['TotalWeight'] = dataclasses.I3Double(frame['I3MCWeightDict']['TotalWeight']*ratio)
    else:
        frame['TotalWeight'] = dataclasses.I3Double(frame['I3MCWeightDict']['TotalWeight'])

tray.Add(glashow_correction)

'''

'''


pol_hadr_0 = '/home/abalagopalv/diffuse/TauStudies/tau_polarization/Hadron_polarization_0.csv'
pol_hadr_minus1='/home/abalagopalv/diffuse/TauStudies/tau_polarization/Hadron_polarization_-1.csv'
pol_lep_0 = '/home/abalagopalv/diffuse/TauStudies/tau_polarization/Lepton_polarization_0.csv'
pol_lep_minus1='/home/abalagopalv/diffuse/TauStudies/tau_polarization/Lepton_polarization_-1.csv'

def tau_polarization(frame):
    leptons = [dataclasses.I3Particle.MuPlus,dataclasses.I3Particle.MuMinus,\
          dataclasses.I3Particle.EMinus, dataclasses.I3Particle.EPlus]
    neutrinos = [dataclasses.I3Particle.NuE, dataclasses.I3Particle.NuEBar,
                     dataclasses.I3Particle.NuMu, dataclasses.I3Particle.NuMuBar,
                     dataclasses.I3Particle.NuTau, dataclasses.I3Particle.NuTauBar]
    from scipy.interpolate import interp1d
    nutype=frame["I3MCWeightDict"]["PrimaryNeutrinoType"]
    inter_type=frame['I3MCWeightDict']['InteractionType']
    had_energy = []
    lep_energy = []
    print(frame['I3EventHeader'].run_id,frame['I3EventHeader'].event_id)
    if (abs(nutype)==16 and inter_type==1.0):
        MCTreeName = 'I3MCTree'
        if frame.Has(MCTreeName):
            for p in frame[MCTreeName].get_primaries():
                    
                if (p.type != dataclasses.I3Particle.NuTau and p.type != dataclasses.I3Particle.NuTauBar):
                    continue
                for c in frame[MCTreeName].children(p):
                    if c.type == dataclasses.I3Particle.TauMinus or c.type == dataclasses.I3Particle.TauPlus:
                        E_Tau = c.energy

                        for d in frame[MCTreeName].get_daughters(c):
                            if d.type == dataclasses.I3Particle.NuTau or d.type == dataclasses.I3Particle.NuTauBar:
                                time_of_int = d.time
                                break
                            else:
                                time_of_int = 0.   #to account for Tau decays outside of detector where theres no daughter neutrino

                        

                        for d in frame[MCTreeName].get_daughters(c):
                            if d.time == time_of_int and d.type not in neutrinos:
                                    if d.type not in leptons:

                                        had_energy.append(d.energy)
                                        
                                    else:
                                        lep_energy.append(d.energy)
                                        
                        y_lep = sum(lep_energy)/E_Tau
                        y_had = sum(had_energy)/E_Tau
                        frame['y_lep'] = dataclasses.I3Double(y_lep)
                        frame['y_had']= dataclasses.I3Double(y_had)
                

    if sum(lep_energy)!=0:
        old_spline=pd.read_csv(pol_lep_0,header=None)
        new_spline=pd.read_csv(pol_lep_minus1,header=None)
        x = old_spline[0]
        y = old_spline[1]
        xn = new_spline[0]
        yn = new_spline[1]
        f1 = interp1d(x, y, kind='cubic')
        f2 = interp1d(xn, yn, kind='cubic')

        num = f2(y_lep)
        denom = f1(y_lep)
        ratio = num/denom
        frame['TotalWeightPol'] = dataclasses.I3Double(frame['TotalWeight'].value*ratio)
        
    elif sum(had_energy)!=0:
        old_spline=pd.read_csv(pol_hadr_0,header=None)
        new_spline=pd.read_csv(pol_hadr_minus1,header=None)
        x = old_spline[0]
        y = old_spline[1]
        xn = new_spline[0]
        yn = new_spline[1]
        f1 = interp1d(x, y, kind='cubic')
        f2 = interp1d(xn, yn, kind='cubic')

        num = f2(y_had)
        denom = f1(y_had)
        ratio = num/denom
        frame['TotalWeightPol'] = dataclasses.I3Double(frame['TotalWeight'].value*ratio)
        
                                

    if not frame.Has('TotalWeightPol'):
        frame['TotalWeightPol'] = dataclasses.I3Double(frame['TotalWeight'].value)

tray.Add(tau_polarization)

### HDF Keys ###
hdfkeys = ['I3EventHeader', 'I3MCWeightDict']
hdfkeys += ['HESEMillipedeFitTruncatedDepositedEnergy', 'HESEMillipedeFitDepositedEnergy']
hdfkeys += ['RecoLogL', 'RecoERatio', 'RecoEConfinement', 'RecoLogE1', 'RecoLogE2', 'RecoLogETot', 'RecoZenith', 'RecoAzimuth', 'RecoL','RecoETot', 'RecoLbyE']
hdfkeys += ['HESEEventclass','MCInteractionEventclass']
hdfkeys += ['HESEMillipedeFitFitParams','MCInteractionDepth']
hdfkeys += ['FinalTopology','DeepCore_Charge','TotalCharge']
hdfkeys += ['ConventionalAtmosphericPassingFractions','PromptAtmosphericPassingFractions']
hdfkeys += ['TrueL', 'TrueLength','TrueETot','TrueE1','TrueE2','TrueZenith', 'TrueAzimuth']
hdfkeys += ['RecoParticle','RecoEnergy','RecoDirection','RecoLength','MCReconstructionEventclass','FinalEventClass']
hdfkeys += ['TotalWeight','TotalWeightPol','y_lep','y_had']
hdfkeys += ['HESETaupedeFitFitParams', 'HESETaupedeFit', 'HESETaupedeFit1', 'HESETaupedeFit2']
hdfkeys += ['HESEMonopodFit_x', 'HESEMonopodFit_y', 'HESEMonopodFit_z']
hdfkeys += ['HESETaupedeFit1_x', 'HESETaupedeFit1_y', 'HESETaupedeFit1_z']
hdfkeys += ['HESETaupedeFit2_x', 'HESETaupedeFit2_y', 'HESETaupedeFit2_z']
hdfkeys += ['MyEgeneratorOutputFrameKey']
hdfkeys += ['FinalTopology_evtgen', 'FinalEventClass_evtgen']

# some keys were missing, especially the SnowStormDict
hdfkeys += [
    'IsHESE_ck',
    'Filenum',
    'TaupedeFitManualFitStatus',
    'issingle',
    'isdouble',
    'istrack',
    'RecoE1',
    'RecoE2',
    'VertexOutgoingHadronNew',
    'VertexOutgoingLeptonNew',
    'TaupedeFitParticles',
    'TaupedeFit',
    'TaupedeFitFitParams',
    'SnowstormParameterDict'
]

outfile_dir = "/data/user/tvaneede/GlobalFit/EventGenerator/hdf/v0/{0}/{1}/{2}/".format(opts.table,opts.folder,opts.dataset)

os.system(f"mkdir -p {outfile_dir}")

outputfile = f"{outfile_dir}/{opts.dataset}_{opts.subfolder}_{opts.topology}.hdf5"

tray.Add(I3HDFWriter, Output=outputfile, Keys=hdfkeys, SubEventStreams=['InIceSplit'])
#tray.Add(I3HDFWriter, Output= filedir+dataset+"_"+opts.topology+".hdf5", Keys=hdfkeys, SubEventStreams=['InIceSplit'])
tray.Execute() 