"""NPZ data adapter. Do NOT pretend unrelated public modalities are paired.
Required arrays: pose [N,T,17,256], imu [N,T,4,6], errors [N,T],
phases [N,T], subjects [N]. Optional severity [N,T] (range 0..1).
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def resample_imu(imu,timestamps,frame_timestamps):
    """Interpolate 100-Hz readings to measured 30-Hz video timestamps."""
    imu=np.asarray(imu,dtype=np.float32)
    result=np.empty((len(frame_timestamps),imu.shape[1],imu.shape[2]),np.float32)
    for m in range(imu.shape[1]):
        for c in range(imu.shape[2]):
            result[:,m,c]=np.interp(frame_timestamps,timestamps,imu[:,m,c])
    return result


def median_filter_imu(x,k=3):
    """Time median filter without scipy dependency; preserves sequence length."""
    a=np.pad(np.asarray(x),((k//2,k//2),(0,0),(0,0)),mode='edge')
    return np.stack([np.median(a[t:t+k],axis=0) for t in range(len(x))]).astype(np.float32)


class MotionDataset(Dataset):
    def __init__(self,path,indices=None):
        arrays=np.load(path,allow_pickle=False)
        required=['pose','imu','errors','phases','subjects']
        missing=[k for k in required if k not in arrays]
        if missing: raise ValueError(f'Missing arrays in {path}: {missing}')
        self.pose=np.asarray(arrays['pose'],np.float32)
        self.imu=np.asarray(arrays['imu'],np.float32)
        self.errors=np.asarray(arrays['errors'],np.int64)
        self.phases=np.asarray(arrays['phases'],np.int64)
        self.subjects=arrays['subjects']
        self.severity=np.asarray(arrays['severity'],np.float32) if 'severity' in arrays else None
        assert self.pose.ndim==4 and self.pose.shape[2:]==(17,256),self.pose.shape
        assert self.imu.shape==(*self.pose.shape[:2],4,6),self.imu.shape
        assert self.errors.shape==self.phases.shape==self.pose.shape[:2]
        self.indices=np.arange(len(self.pose)) if indices is None else np.asarray(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self,index):
        i=self.indices[index]
        item={'pose':torch.from_numpy(self.pose[i]),'imu':torch.from_numpy(self.imu[i]),
              'errors':torch.tensor(self.errors[i]),'phases':torch.tensor(self.phases[i])}
        if self.severity is not None: item['severity']=torch.from_numpy(self.severity[i])
        return item


def subject_split(subjects,seed=42,ratios=(0.7,0.1,0.2)):
    """Partition by subject ID rather than clip to prevent identity leakage."""
    subjects=np.asarray(subjects)
    ids=np.unique(subjects)
    if len(ids)<3: raise ValueError('At least three subjects are required for three splits')
    rng=np.random.default_rng(seed);rng.shuffle(ids)
    ntr=max(1,int(round(len(ids)*ratios[0])))
    nv=max(1,int(round(len(ids)*ratios[1])))
    if ntr+nv>=len(ids): ntr,nv=len(ids)-2,1
    parts=[ids[:ntr],ids[ntr:ntr+nv],ids[ntr+nv:]]
    return tuple(np.flatnonzero(np.isin(subjects,x)) for x in parts)


def generate_synthetic(path,n=36,t=64,seed=42):
    """Synthetic data ONLY for installation/smoke tests, never for publication metrics."""
    rng=np.random.default_rng(seed)
    subjects=np.repeat(np.arange(max(3,n//3)),3)[:n]
    n=len(subjects)
    phase=np.tile(np.arange(t)//(t//4), (n,1)).clip(0,3).astype(np.int64)
    errors=rng.integers(0,5,(n,t),dtype=np.int64)
    latent=np.sin(np.linspace(0,4*np.pi,t))[None,:,None,None]
    pose=rng.normal(0,.5,(n,t,17,256)).astype(np.float32)
    pose+=latent.astype(np.float32)*.3
    imu=rng.normal(0,.5,(n,t,4,6)).astype(np.float32)
    imu+=latent.astype(np.float32)*.4
    np.savez_compressed(path,pose=pose,imu=imu,errors=errors,phases=phase,subjects=subjects)
