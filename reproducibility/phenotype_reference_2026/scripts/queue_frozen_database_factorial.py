#!/usr/bin/env python3
"""Run controlled E2 arms after the active CCRE arm, using four workers total."""
from datetime import datetime,timezone
import json,os,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/database_reference_factorial_20261111'


def write(state):
    state['updated_utc']=datetime.now(timezone.utc).isoformat()
    p=OUT/'QUEUE_STATUS.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(state,indent=2)+'\n');q.replace(p)


def active_ccre():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except (OSError,PermissionError):continue
        if any(a.endswith(b'run_frozen_database_factor.py') for a in args) and any(a.endswith(b'/CCRE_Jan') for a in args):return True
    return False


def main():
    write({'status':'WAITING_FOR_CCRE_JAN_CHECKPOINT_RUN'})
    status=OUT/'CCRE_Jan/status.json'
    while True:
        if status.exists() and json.loads(status.read_text())['status'].startswith('COMPLETE_'):break
        if not active_ccre():
            write({'status':'BLOCKED_CCRE_PROCESS_STOPPED_WITHOUT_COMPLETE_STATUS'});raise SystemExit(1)
        time.sleep(5)
    gate=json.loads((OUT/'E2_SOURCE_GATE.json').read_text())
    if gate['status']!='PASS_ALL525_EXACT_ORIGINAL_DNA_BYTES':raise ValueError('E2 source gate incomplete')
    jobs=[];env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1'
    for label,version in [('Jan','2026-01-21.1'),('Aug','2026-08-07.1')]:
        target=OUT/('E2_'+label);final=target/'status.json'
        if final.exists() and json.loads(final.read_text())['status'].startswith('COMPLETE_'):continue
        argv=['python','-u','scripts/run_frozen_database_factor.py',
          '--manifest',str(OUT/'E2_execution_manifest.csv'),'--fasta-dir',str(OUT/'E2_recovery/fastas'),
          '--model',str(ROOT/'results/frozen_model_challenge_20261110/received/project/checkpoints/external_validation_file11c/file11c_final_model.joblib'),
          '--amrfinder',str(ROOT/'.tools/amrfinder427/bin/amrfinder'),'--database',str(ROOT/'.tools/amrfinder_db'/version),
          '--expected-db',version,'--label-column','historical_label','--out',str(target),'--workers','4']
        if target.exists():argv.append('--resume')
        log=(OUT/('E2_'+label+'.log')).open('a')
        child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((label,child,log))
        write({'status':'RUNNING_E2_DATABASE_ARM','arm':label,'pid':child.pid,'workers':4})
        while child.poll() is None:time.sleep(5)
        if child.returncode!=0:
            write({'status':'FAILED_E2_ARM','arm':label,'exit_code':child.returncode});raise SystemExit(1)
    write({'status':'RUNNING_E2_DATABASE_ARMS','processes':{n:p.pid for n,p,_ in jobs},'workers_per_arm':4})
    while any(p.poll() is None for _,p,_ in jobs):time.sleep(5)
    failed={n:p.returncode for n,p,_ in jobs if p.returncode!=0}
    for _,_,log in jobs:log.close()
    write({'status':'COMPLETE_CONTROLLED_CALLER_ARMS' if not failed else 'FAILED_E2_ARMS_CHECKPOINTS_RETAINED','failed':failed})
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
