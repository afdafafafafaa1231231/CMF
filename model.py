"""CMF-ErrNet reproduction from the supplied manuscript.
Pose input is pre-extracted HRNet joint features [B,T,17,256].
IMU input is synchronized 30-Hz [B,T,4,6].
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class ResidualTCN(nn.Module):
    def __init__(self, channels=128, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
    def forward(self, x):
        z = F.relu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        return F.relu(x + z)


class PoseEncoder(nn.Module):
    """Joint-wise temporal depthwise convolution (9-frame window).
    The paper calls this DepthwiseConv3D; for joint vectors, a grouped 1D
    temporal operation is an explicit implementation choice.
    """
    def __init__(self, joints=17, features=256):
        super().__init__()
        self.joints, self.features = joints, features
        self.temporal = nn.Conv1d(joints * features, joints * features, 9,
                                  padding=4, groups=joints * features)
        self.norm = nn.LayerNorm(features)
    def forward(self, x):
        b,t,j,d = x.shape
        assert (j,d) == (self.joints,self.features)
        y = x.reshape(b,t,j*d).transpose(1,2)
        y = self.temporal(y).transpose(1,2).reshape(b,t,j,d)
        return self.norm(x+y)


class IMUEncoder(nn.Module):
    def __init__(self, sensors=4, channels=6, dim=128):
        super().__init__()
        self.sensors = sensors
        self.input = nn.Linear(channels,dim)
        self.blocks = nn.Sequential(*(ResidualTCN(dim,d) for d in (1,2,4,8)))
        self.norm = nn.LayerNorm(dim)
    def forward(self,x):
        b,t,m,c=x.shape
        assert m==self.sensors and c==6
        y = self.input(x).permute(0,2,3,1).reshape(b*m,-1,t)
        y = self.blocks(y).reshape(b,m,-1,t).permute(0,3,1,2)
        return self.norm(y)


class CrossModalAttention(nn.Module):
    """Symmetric 4-head joint-sensor attention; returns [B,T,17,128]."""
    def __init__(self, pose_dim=256, imu_dim=128, d=128, heads=4):
        super().__init__()
        self.pose_projection = nn.Linear(pose_dim,d)
        self.imu_projection = nn.Linear(imu_dim,d)
        self.pose_to_imu = nn.MultiheadAttention(d,heads,batch_first=True)
        self.imu_to_pose = nn.MultiheadAttention(d,heads,batch_first=True)
        self.norm_p = nn.LayerNorm(d)
        self.norm_i = nn.LayerNorm(d)
    def forward(self,p,i,enabled=True):
        b,t,j,_=p.shape
        m=i.shape[2]
        p=self.pose_projection(p)
        i=self.imu_projection(i)
        if not enabled:
            return p,i,None
        pq=p.reshape(b*t,j,-1)
        iq=i.reshape(b*t,m,-1)
        p_update,weights=self.pose_to_imu(pq,iq,iq,need_weights=True)
        i_update,_=self.imu_to_pose(iq,pq,pq,need_weights=False)
        p=self.norm_p(pq+p_update).reshape(b,t,j,-1)
        i=self.norm_i(iq+i_update).reshape(b,t,m,-1)
        return p,i,weights.reshape(b,t,j,m)


class CMFErrNet(nn.Module):
    def __init__(self,num_errors=5,num_phases=4,pose_dim=256, hidden=256,
                 modality_dropout=0.2, ablation='full'):
        super().__init__()
        self.ablation=ablation
        self.modality_dropout=modality_dropout
        self.pose=PoseEncoder(features=pose_dim)
        self.imu=IMUEncoder()
        self.cross=CrossModalAttention(pose_dim=pose_dim)
        self.context=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,2))
        self.temporal=nn.LSTM(128,hidden,num_layers=2,batch_first=True,
                              bidirectional=True,dropout=0.3)
        self.output_dropout=nn.Dropout(0.3)
        self.phase_head=nn.Linear(hidden*2,num_phases)
        self.phase_embedding=nn.Embedding(num_phases,32)
        self.error_head=nn.Sequential(nn.Linear(hidden*2+32,256),nn.ReLU(),
                                      nn.Linear(256,num_errors))
        self.no_phase_error_head=nn.Linear(hidden*2,num_errors)
        self.severity_head=nn.Sequential(nn.Linear(hidden*2,1),nn.Sigmoid())
    def forward(self,pose,imu,return_attention=False):
        p=self.pose(pose)
        i=self.imu(imu)
        p,i,attn=self.cross(p,i,enabled=self.ablation not in ('no_attention','pose_only'))
        pv=p.mean(dim=2)
        iv=i.mean(dim=2)
        if self.training and self.modality_dropout>0 and self.ablation!='pose_only':
            # Entire-modality dropout per clip; gated pooling learns fallbacks.
            b=pose.shape[0]
            apply=torch.rand(b,device=pose.device)<self.modality_dropout
            drop_imu=torch.rand(b,device=pose.device)<0.5
            pv=pv.masked_fill((apply & ~drop_imu)[:,None,None],0)
            iv=iv.masked_fill((apply & drop_imu)[:,None,None],0)
        if self.ablation=='pose_only':
            gate=torch.stack((torch.ones_like(pv[...,0]),torch.zeros_like(pv[...,0])),dim=-1)
        elif self.ablation=='no_gate':
            gate=torch.full((*pv.shape[:2],2),0.5,device=pv.device,dtype=pv.dtype)
        else:
            gate=F.softmax(self.context(torch.cat((pv,iv),dim=-1)),dim=-1)
        z=gate[...,0:1]*pv + gate[...,1:2]*iv
        h,_=self.temporal(z)
        h=self.output_dropout(h)
        phase_logits=self.phase_head(h)
        if self.ablation=='no_phase':
            err=self.no_phase_error_head(h)
        else:
            # Soft phase embedding keeps gradients to phase classifier.
            softphase=F.softmax(phase_logits,dim=-1)
            emb=softphase @ self.phase_embedding.weight
            err=self.error_head(torch.cat((h,emb),dim=-1))
        out={'error_logits':err,'phase_logits':phase_logits,
             'severity':self.severity_head(h).squeeze(-1),'gate':gate}
        if return_attention: out['attention']=attn
        return out


def compute_loss(out,labels,phases,severity=None,valid=None,
                 phase_weight=0.3,smooth_weight=0.1,severity_weight=0.0):
    if valid is None: valid=torch.ones_like(labels,dtype=torch.bool)
    if not valid.any(): raise ValueError('A batch contains no valid frames')
    cls=F.cross_entropy(out['error_logits'][valid],labels[valid])
    phase=F.cross_entropy(out['phase_logits'][valid],phases[valid])
    prob=F.softmax(out['error_logits'],dim=-1)
    pair=valid[:,1:] & valid[:,:-1]
    smooth=(prob[:,1:]-prob[:,:-1]).abs().sum(-1)
    smooth=smooth[pair].mean() if pair.any() else cls.new_zeros(())
    loss=cls+phase_weight*phase+smooth_weight*smooth
    stats={'classification':cls.item(),'phase':phase.item(),'smooth':smooth.item()}
    if severity is not None and severity_weight>0:
        sl=F.mse_loss(out['severity'][valid],severity[valid].float())
        loss=loss+severity_weight*sl
        stats['severity']=sl.item()
    return loss,stats
