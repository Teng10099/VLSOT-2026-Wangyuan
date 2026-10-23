"""Package exact model weights, preserving the inference contract and tokenizer."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--source-project',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    import torch
    target=Path(a.output);target.mkdir(parents=True,exist_ok=False)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    state=ck.get('model_state',ck)
    state={k[7:] if k.startswith('module.') else k:v for k,v in state.items()}
    weight=target/'model.pth'
    packaged={'model_state':state,'epoch':ck.get('epoch')}
    for name in ('target_encoding','model_options'):
        if name in ck: packaged[name]=ck[name]
    if any('official_geometry_head' in k for k in state):
        if packaged.get('target_encoding') != 'camera_geometric_center_v2' or not isinstance(packaged.get('model_options'),dict):
            raise RuntimeError('V2 weights require target_encoding and model_options')
    torch.save(packaged,weight)
    loaded=torch.load(weight,map_location='cpu',weights_only=True,mmap=True)
    reloaded=loaded['model_state']
    if set(reloaded)!=set(state) or any(not torch.equal(state[k],reloaded[k]) for k in state):
        raise RuntimeError('Packaged state_dict is not bitwise identical')
    for name in ('target_encoding','model_options'):
        if loaded.get(name) != ck.get(name):
            raise RuntimeError('Packaged inference contract changed: '+name)
    tokenizer=target/'roberta-base';tokenizer.mkdir()
    for name in ('config.json','tokenizer.json','vocab.json','dict.txt'):
        shutil.copyfile(Path(a.source_project)/'roberta-base'/name,tokenizer/name)
    manifest={'model_state_keys':len(state),'checkpoint_epoch':ck.get('epoch'),'state_dict_bitwise_verified':True,'training_optimizer_omitted':True,'target_encoding':packaged.get('target_encoding'),'model_options':packaged.get('model_options'),'source_checkpoint_sha256':digest(Path(a.checkpoint)),'training_provenance':'See corresponding training run manifest; inherited checkpoint history is not independently verified','files':[]}
    for f in sorted(target.rglob('*')):
        if f.is_file():manifest['files'].append({'path':str(f.relative_to(target)),'bytes':f.stat().st_size,'sha256':digest(f)})
    (target/'weights_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
