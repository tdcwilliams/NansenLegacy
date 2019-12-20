#! /usr/bin/env python
import sys
import numpy as np
import argparse
import subprocess
import json

def parse_args(cli):
    parser = argparse.ArgumentParser("cancel ensemble jobs")
    parser.add_argument('job_string', type=str,
            help="cancel jobs with this string in the jobname")
    parser.add_argument('user', type=str,
            help="cancel jobs from this user")
    return parser.parse_args(cli)

def get_job_ids(args):
    n_name = len(args.job_string) + 5
    fmt = ('%'+ '.%ij' %n_name) + ' %i'
    queue_cmd = ['squeue', '-o', fmt] # give job names
    queue_cmd += ['-u', args.user]
    running_jobs = subprocess.run(
                    queue_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    )
    sp = running_jobs.stdout.split()[2:]
    arr = np.array(sp).reshape((int(len(sp)/2),2))
    jobs_to_cancel = []
    for name, jobid in arr:
        if args.job_string in name:
            jobs_to_cancel.append((name, jobid))
    return jobs_to_cancel

def cancel_jobs(jobs_to_cancel):
    print(json.dumps(
        jobs_to_cancel, indent=4))
    for name, jobid in jobs_to_cancel:
        cmd = ['scancel', jobid]
        print(' '.join(cmd))
        #subprocess.run(cmd)

args = parse_args(sys.argv[1:])
cancel_jobs(get_job_ids(args))
