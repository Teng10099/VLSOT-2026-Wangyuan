"""Compare temporal modes ONLY on a whitelisted training clip; never leaderboard GT."""
import argparse
import json
from pathlib import Path
import numpy as np
from geometry_decode import decode_native_predictions
from batch_infer import OFFICIAL_IDS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-split',required=True)
    p.add_argument('--sequence-json',required=True)
    p.add_argument('--carry-dir',required=True)
    p.add_argument('--reset-dir',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    video_id=Path(a.sequence_json).stem
    train=set(Path(a.training_split).read_text().split())
    if video_id not in train or video_id in OFFICIAL_IDS:
        raise ValueError('This utility only evaluates an explicit non-test training ID')
    obj=json.loads(Path(a.sequence_json).read_text());obj=obj[0] if isinstance(obj,list) else obj
    truth_lwh=np.asarray([obj['frame'+str(i)][10:13] for i in range(30)],dtype=float)
    truth_xyz=np.asarray([obj['frame'+str(i)][13:16] for i in range(30)],dtype=float)
    records=[]
    for mode,directory in [('carry',a.carry_dir),('reset_each_frame',a.reset_dir)]:
        d=json.loads((Path(directory)/'diagnostics.json').read_text())
        if d['video_id']!=video_id or d['temporal_mode']!=mode: raise ValueError('Mismatched diagnostic run')
        pred_lwh=[];pred_xyz=[]
        for frame in d['frames']:
            heads={k:np.asarray(frame[k]) for k in ['pred_logits','pred_boxes','pred_3d_dim','pred_depth']}
            dec=decode_native_predictions(heads,frame['source_size_wh'],None,np.zeros((heads['pred_logits'].shape[-1],3)),center_encoding='legacy_native_xy',input_sizes_wh=frame['input_size_wh'],source_from_input=frame['source_from_input'])
            pred_lwh.append(dec['dimensions_lwh'][0]);pred_xyz.append(dec['label_position_native_xyz'][0])
        distances=np.linalg.norm(np.asarray(pred_xyz)-truth_xyz,axis=1)
        records.append({'mode':mode,'frames':30,'mean_center_distance_m':float(distances.mean()),'median_center_distance_m':float(np.median(distances)),'dimension_mae_lwh_m':np.abs(np.asarray(pred_lwh)-truth_lwh).mean(axis=0).tolist(),'forward_mean_seconds':float(np.mean([f['forward_seconds'] for f in d['frames']]))})
    result={'purpose':'Training-set runtime ablation; NOT a held-out evaluation or official score','video_id':video_id,'checkpoint_identical':True,'seed_per_video':444,'metric':'Euclidean distance between predicted and source label XYZ in infrastructure virtual LiDAR coordinates; mean absolute dimension errors','limitations':['One training clip (30 correlated frames), descriptive only; no generalization claim.','Weights were provided by a third party; complete training provenance and disjointness were not independently established.'],'results':records}
    target=Path(a.output)
    if target.exists(): raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
