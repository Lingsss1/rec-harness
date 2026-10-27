import os, importlib.util, torch
HERE=os.path.dirname(os.path.abspath(__file__))
DATASETS=['Movies_and_TV','Industrial_and_Scientific','Electronics','CDs_and_Vinyl']
_models={}
_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_spec=importlib.util.spec_from_file_location('duorec_model_def',os.path.join(HERE,'model_def.py'))
_mod=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
def predict(user_id,history,candidates,*,dataset_name=None):
    ds=DATASETS[0] if dataset_name is None else dataset_name
    if ds not in DATASETS: raise ValueError('Unknown dataset: '+str(ds))
    if not candidates: return []
    if ds not in _models:
        ck=torch.load(os.path.join(HERE,ds+'_model.pt'),map_location='cpu',weights_only=True)
        model=_mod.Model(**ck['config']); model.load_state_dict(ck['state']); model.to(_device).eval()
        _models[ds]=model
    model=_models[ds]; hist=list(history[-model.maxlen:]); seq=[0]*(model.maxlen-len(hist))+hist
    with torch.inference_mode():
        return model.predict(torch.tensor([seq],dtype=torch.long,device=_device),torch.tensor([candidates],dtype=torch.long,device=_device))[0].cpu().tolist()
