import os,tempfile
import torch
from model import CMFErrNet,compute_loss
from data import generate_synthetic,MotionDataset,subject_split,resample_imu,median_filter_imu
import numpy as np

torch.manual_seed(7)
model=CMFErrNet(num_errors=5)
p=torch.randn(2,16,17,256)
i=torch.randn(2,16,4,6)
out=model(p,i,return_attention=True)
assert out['error_logits'].shape==(2,16,5)
assert out['phase_logits'].shape==(2,16,4)
assert out['severity'].shape==(2,16)
assert out['attention'].shape==(2,16,17,4)
assert torch.allclose(out['gate'].sum(-1),torch.ones(2,16),atol=1e-5)
loss,_=compute_loss(out,torch.randint(0,5,(2,16)),torch.randint(0,4,(2,16)))
loss.backward()
assert torch.isfinite(loss)
assert any(p.grad is not None for p in model.parameters())
for variant in ('no_gate','no_attention','no_phase','pose_only'):
    with torch.no_grad():
        m=CMFErrNet(ablation=variant)
        z=m(p[:1,:4],i[:1,:4]);assert z['error_logits'].shape==(1,4,5)
with tempfile.TemporaryDirectory() as d:
    path=os.path.join(d,'test.npz');generate_synthetic(path,n=12,t=16)
    ds=MotionDataset(path);tr,va,te=subject_split(ds.subjects)
    s=[set(ds.subjects[x].tolist()) for x in (tr,va,te)]
    assert not s[0]&s[1] and not s[0]&s[2] and not s[1]&s[2]
    assert sum(map(len,(tr,va,te)))==len(ds)
    assert ds[0]['pose'].shape==(16,17,256)
raw=np.random.default_rng(0).normal(size=(30,4,6)).astype(np.float32)
assert resample_imu(raw,np.linspace(0,1,30),np.linspace(0,1,10)).shape==(10,4,6)
assert median_filter_imu(raw).shape==raw.shape
print('PASS: shapes, attention, normalization, backward, ablations, dataset, subject split, IMU preprocessing')
