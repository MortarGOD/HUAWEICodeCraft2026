"""Complete preregistered v434 screening and serial phase profiling."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_attention_campaign_followup import command


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int,default=0);args=parser.parse_args()
    if args.wait_pid:
        while True:
            try:os.kill(args.wait_pid,0)
            except ProcessLookupError:break
            time.sleep(5)
    assert json.loads((ROOT/'docs/v433-attention-block_resample.json').read_text())['completed']
    assert json.loads((ROOT/'docs/v434-attention-mha.json').read_text())['completed']
    for tag,bank,extra in (
        ('paired','local_fitted_models_attention_feedback_v2',['--groups','0,10,20,30,40,49']),
        ('covariance','local_fitted_models_covariance_v1',['--groups','0,10,20,30,40,49']),
        ('row_permute','local_fitted_models_attention_feedback_v2',['--groups','0,24,49','--isolated','--decoy-mode','row_permute']),
        ('block_resample','local_fitted_models_attention_feedback_v2',['--groups','0,24,49','--isolated','--decoy-mode','block_resample'])):
        stem=f'docs/v434-attention-{tag}'
        command(['tools/verify_attention_campaign.py','--candidate','434','--dataset',bank,'--output',stem+'.json',*extra],stem+'.log')
    command(['tools/profile_attention_campaign_20260908.py'],'docs/attention-campaign-serial-profile.log')
    command(['tools/summarize_attention_campaign.py'],'docs/attention-campaign-summary.log')
    print('FINAL SCREEN AND PROFILE DONE',flush=True)


if __name__=='__main__':main()
