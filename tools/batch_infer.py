"""Deterministic VLSOT clip inference using images, text and calibration only."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

OFFICIAL_IDS = ('000237','001483','003235','006543','024386','033447','035729','036120','047771','051657')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('config','source-project','checkpoint','data-root','output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--ids', default=','.join(OFFICIAL_IDS))
    p.add_argument('--gpu', default='1')
    p.add_argument('--temporal-mode', choices=('carry','reset_each_frame'), default='carry')
    args=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    project=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(project))
    import numpy as np
    import torch
    import yaml
    from PIL import Image
    from tools.smoke_infer import read_sequence, load_projection
    from lib.datasets.mono3drefer.mono3drefer_utils import get_affine_transform
    from lib.helpers.model_helper import build_model
    ids=args.ids.split(',')
    if len(ids)!=len(set(ids)) or any(len(x)!=6 or not x.isdigit() for x in ids):
        raise ValueError('IDs must be unique six-digit strings')
    root=Path(args.data_root).resolve()
    index={x:[] for x in ids}
    for path in (root/'video').rglob('*.json'):
        if path.stem in index: index[path.stem].append(path)
    if any(len(v)!=1 for v in index.values()):
        raise ValueError('Missing or ambiguous requested sequence paths')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    cfg['model'].update(text_encoder_path=str(Path(args.source_project)/'roberta-base'),text_encoder_from_config=True,swin_use_pretrained_backbone=False,device='cpu')
    cfg['model'].pop('pretrained_backbone_path',None)
    seed=int(cfg.get('random_seed',444));torch.set_num_threads(4);torch.manual_seed(seed)
    started=time.perf_counter()
    model,criterion=build_model(cfg['model']);del criterion
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    state=ckpt.get('model_state',ckpt)
    state={k[7:] if k.startswith('module.') else k:v for k,v in state.items()}
    model.load_state_dict(state,strict=True);model.cuda().eval()
    decoder=model.mono3dvg_transformer.decoder
    def reset_memory():
        decoder.memory_tokens=None
        decoder.TTM_model.simpleDNC.current_flag=0
        decoder.TTM_model.simpleDNC.split_memory_tokens=[]
    resolution=np.asarray([1920,1080],dtype=np.int32)
    mean=np.asarray([.485,.456,.406],dtype=np.float32)
    std=np.asarray([.229,.224,.225],dtype=np.float32)
    transform_manifest={'schema_version':1,'source_coordinate_system':'infrastructure_virtual_lidar','target_coordinate_system':'virtual_world','source_verification':'Training sequence 039373, track 004049, frame 007823: customized sequence dimensions and XYZ exactly match original image-timestamp-aligned label/camera object. Official V2X-Seq documentation identifies these as infrastructure virtual LiDAR coordinates. No test target numeric fields are read for inference or coordinate selection.','sequences':{}}
    all_summaries=[]
    print('MODEL_LOADED_STRICT',len(state),flush=True)
    with torch.inference_mode():
        for video_id in ids:
            reset_memory();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
            torch.cuda.reset_peak_memory_stats()
            clip_start=time.perf_counter()
            actual,description,frames=read_sequence(index[video_id][0])
            if actual!=video_id: raise ValueError('Sequence ID mismatch')
            clip=out/video_id;clip.mkdir()
            summary={'status':'raw_inference_complete','platform_submission_ready':False,'coordinate_system':'encoded_native','video_id':video_id,'checkpoint_epoch':ckpt.get('epoch'),'state_dict_keys':len(state),'state_dict_load':'strict_success','seed_per_video':seed,'temporal_mode':args.temporal_mode,'input_text':description,'frames':[]}
            arrays={};world=[]
            for frame in frames:
                if args.temporal_mode=='reset_each_frame': reset_memory()
                image_path=root/'img'/frame['image_name']
                calib_path=root/'calib/camera_intrinsic'/frame['calibration_name']
                projection=load_projection(calib_path,np)
                with Image.open(image_path) as im:
                    im=im.convert('RGB');size=np.asarray(im.size,dtype=np.float32)
                    _,inverse=get_affine_transform(size/2,size,0,resolution,inv=1)
                    im=im.transform(tuple(resolution.tolist()),Image.AFFINE,tuple(inverse.reshape(-1).tolist()),Image.BILINEAR)
                    pixels=np.asarray(im,dtype=np.float32)/255.
                pixels=np.ascontiguousarray(((pixels-mean)/std).transpose(2,0,1))
                torch.cuda.synchronize();before=time.perf_counter()
                prediction=model(torch.from_numpy(pixels).unsqueeze(0).cuda(),torch.from_numpy(projection).unsqueeze(0).cuda(),torch.from_numpy(size).unsqueeze(0).cuda(),[description])
                torch.cuda.synchronize()
                record=dict(frame,source_size_wh=size.tolist(),input_size_wh=resolution.tolist(),source_from_input=inverse.tolist(),projection_matrix=projection.tolist(),forward_seconds=time.perf_counter()-before)
                for key in ('pred_logits','pred_boxes','pred_3d_dim','pred_depth','pred_angle'):
                    val=prediction[key].detach().float().cpu().numpy()
                    if not np.isfinite(val).all(): raise ValueError('Nonfinite predictions')
                    arrays['frame_{:02d}_{}'.format(frame['frame_index'],key)]=val;record[key]=val.tolist()
                summary['frames'].append(record)
                wp=root/'calib/virtuallidar_to_world'/Path(frame['image_name']).with_suffix('.json').name
                raw=wp.read_bytes();trans=json.loads(raw)
                world.append(dict(frame_index=frame['frame_index'],image_name=frame['image_name'],rotation=trans['rotation'],translation=trans['translation'],calibration_path=str(wp),calibration_sha256=hashlib.sha256(raw).hexdigest()))
            summary['elapsed_seconds']=time.perf_counter()-clip_start
            summary['peak_cuda_memory_mib']=torch.cuda.max_memory_allocated()/1024**2
            summary['raw_dimensions_all_positive']=all(bool((v>0).all()) for k,v in arrays.items() if k.endswith('pred_3d_dim'))
            np.savez_compressed(clip/'raw_outputs.npz',**arrays)
            (clip/'diagnostics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
            transform_manifest['sequences'][video_id]=world
            all_summaries.append({k:summary[k] for k in ('video_id','elapsed_seconds','peak_cuda_memory_mib','raw_dimensions_all_positive')})
            print('CLIP_COMPLETE',json.dumps(all_summaries[-1]),flush=True)
    (out/'world_transforms.json').write_text(json.dumps(transform_manifest,indent=2),encoding='utf-8')
    (out/'batch_summary.json').write_text(json.dumps(dict(clips=all_summaries,total_seconds=time.perf_counter()-started,checkpoint_epoch=ckpt.get('epoch'),torch=torch.__version__,gpu=torch.cuda.get_device_name(0),temporal_mode=args.temporal_mode),indent=2),encoding='utf-8')
    print('BATCH_COMPLETE',str(out),flush=True)

if __name__=='__main__': main()
