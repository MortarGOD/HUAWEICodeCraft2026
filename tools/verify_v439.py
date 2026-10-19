"""Post-push whole-Attention comparisons for the frozen reference transplant."""
from argparse import Namespace
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.verify_attention_campaign import run

if __name__=='__main__':
    for name,bank,groups,isolated,decoy in (
        ('paired','local_fitted_models_attention_feedback_v2','0,8,17,25,34,49',False,'row_permute'),
        ('covariance','local_fitted_models_covariance_v1','0,8,17,25,34,49',False,'row_permute'),
        ('row_permute','local_fitted_models_attention_feedback_v2','0,24,49',True,'row_permute'),
        ('block_resample','local_fitted_models_attention_feedback_v2','0,24,49',True,'block_resample')):
        run(Namespace(parent=435,candidate=439,dataset=bank,groups=groups,isolated=isolated,
                      decoy_mode=decoy,output=f'docs/v439-attention-{name}.json'))
