import sys, os, glob


# set the inputs
reco_version = "v0"

reconstruction_type  = "RecowithBfr"
simulation_type      = "Baseline" # Perturbed
simulation_dataset   = "22086" # also 22085 for nutau
simulation_subfolder = "0000000-0000999"
simulation_flavor    = "NuTau"

# fixed paths
work_path = "/data/user/tvaneede/GlobalFit/EventGenerator"
reco_input_path = f"/data/ana/Diffuse/GlobalFit_Flavor/taupede/SnowStorm/{reconstruction_type}/{simulation_type}/{simulation_dataset}/{simulation_subfolder}"

reco_out_path = f"{work_path}/reco_files/{reconstruction_type}/{simulation_type}/{simulation_dataset}/{simulation_subfolder}"

# fixed dag paths
dag_base_path = "/scratch/tvaneede/reco/evt_gen"
dag_name = f"reco_dag_{reco_version}_{simulation_type}_{simulation_dataset}_{simulation_subfolder}"

dag_path      = f"{dag_base_path}/{dag_name}"
log_dir       = f"{dag_path}/logs"
backup_path   = f"{work_path}/backup_scripts/{reco_version}"

# creating folders and copying scripts
print("creating", dag_path)
os.system(f"mkdir -p {dag_path}")
os.system(f"mkdir -p {log_dir}")
os.system(f"mkdir -p {reco_out_path}")
os.system(f"mkdir -p {backup_path}")
os.system(f"cp reco.sub {dag_path}")

# backup scripts
os.system(f"cp {work_path}/dag/reco.sub {backup_path}")
os.system(f"cp {work_path}/dag/wrapper.sh {backup_path}")
os.system(f"cp {work_path}/run_evt_gen.py {backup_path}")
os.system(f"cp {work_path}/run_evt_gen.sh {backup_path}")


# create the dag job
outfile = open(f"{dag_path}/submit.dag", 'w')

infiles_list = glob.glob(f"{reco_input_path}/Reco_{simulation_flavor}_*.i3.bz2")
print(f"found {len(infiles_list)} files")

# for i in range(job_low, job_high+1):
for INFILES in infiles_list:

    filename = os.path.basename(INFILES)
    JOBID = filename.split("_")[2] # gives the run number
    OUTFILE = f"{reco_out_path}/Reco_{simulation_flavor}_{JOBID}_out.i3.bz2"

    print(INFILES)
    print(filename, JOBID)
    print(OUTFILE)

    outfile.write(f"JOB {JOBID} reco.sub\n")
    outfile.write(f'VARS {JOBID} LOGDIR="{log_dir}"\n')
    outfile.write(f'VARS {JOBID} JOBID="{JOBID}"\n')
    outfile.write(f'VARS {JOBID} INFILES="{INFILES}"\n')
    outfile.write(f'VARS {JOBID} OUTFILE="{OUTFILE}"\n')

    break
