
import os

runs = {
    "NuTau" : ["22085", "22086"],
    "NuMu"  : ["22043", "22044"],
    "NuE"   : ["22082", "22083"],
}

base_path =  "/data/user/tvaneede/GlobalFit/EventGenerator"
this_path = f"{base_path}/create_hdf"

def get_job_list( flavor = "NuTau", reco_type = "RecowithBfr", simulation_type = "Baseline" ):

    jobs = {}

    directory = f"/data/user/tvaneede/GlobalFit/EventGenerator/reco_files/{reco_type}/{simulation_type}"

    for run in runs[flavor]:
        jobs[run] = {}

        directory_run = f"{directory}/{run}"

        folders = [f for f in os.listdir(directory_run) if os.path.isdir(os.path.join(directory_run, f))]
        for folder in folders:
            jobs[run][folder] = os.path.join(directory_run, folder)

        print("run", run, "folders", len(folders))

    return jobs

###
### Topology DoubleCascades
###    
reco_type = "RecowithBfr"
simulation_type = "Baseline"
dag_version = "v0"

# fixed dag paths
dag_path = f"/scratch/tvaneede/create_hdf_files_evtgen/{dag_version}/snowstorm_{reco_type}_{simulation_type}"
log_dir       = f"{dag_path}/logs"

# creating folders and copying scripts
print("creating", dag_path)
os.system(f"mkdir -p {dag_path}")
os.system(f"mkdir -p {log_dir}")
os.system(f"cp snowstorm.sub {dag_path}")

outfile = open(f"{dag_path}/submit.dag", 'w')

# for topology in ["DoubleCascades", "Cascades", "Tracks"]:
for topology in ["DoubleCascades"]:

    for flavor in runs:

        jobs = get_job_list( flavor = flavor, reco_type = reco_type, simulation_type = simulation_type )
        for run in jobs:
            for folder in jobs[run]:
                # cmd = f"python {base_path}/FullPID_NNMFit_hdf.py --Topology {topology} --Table {reco_type} --Folder {simulation_type} --Dataset {run} --subfolder {folder}"
                # cmd = f"{this_path}/snowstorm.sh --Topology {topology} --Table {reco_type} --Folder {simulation_type} --Dataset {run} --subfolder {folder}"
                # print(cmd)
                # # os.system(cmd)

                jobid = f"hdf_{topology}_{flavor}_{run}_{folder}"
                outfile.write(f"JOB {jobid} snowstorm.sub\n")
                outfile.write(f'VARS {jobid} LOGDIR="{log_dir}"\n')
                outfile.write(f'VARS {jobid} JOBID="{jobid}"\n')
                outfile.write(f'VARS {jobid} Topology="{topology}"\n')
                outfile.write(f'VARS {jobid} Table="{reco_type}"\n')
                outfile.write(f'VARS {jobid} Folder="{simulation_type}"\n')
                outfile.write(f'VARS {jobid} Dataset="{run}"\n')
                outfile.write(f'VARS {jobid} subfolder="{folder}"\n')
