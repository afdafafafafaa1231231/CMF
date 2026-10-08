import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import CMFErrNet,compute_loss
from data import MotionDataset,subject_split,generate_synthetic


def metrics(model,loader,device,nclass):
    model.eval();ys=[];ps=[];phase_correct=0;frames=0;total_loss=0
    with torch.no_grad():
        for b in loader:
            p=b['pose'].to(device);i=b['imu'].to(device)
            y=b['errors'].to(device);ph=b['phases'].to(device)
            out=model(p,i)
            total_loss+=torch.nn.functional.cross_entropy(out['error_logits'].flatten(0,1),y.flatten()).item()*y.numel()
            pred=out['error_logits'].argmax(-1)
            ys.extend(y.cpu().reshape(-1).tolist());ps.extend(pred.cpu().reshape(-1).tolist())
            phase_correct+=(out['phase_logits'].argmax(-1)==ph).sum().item()
            frames+=y.numel()
    ys=np.asarray(ys);ps=np.asarray(ps)
    if not frames: return {}
    f1=[];precision=[];recall=[]
    for k in range(nclass):
        tp=np.sum((ys==k)&(ps==k));fp=np.sum((ys!=k)&(ps==k));fn=np.sum((ys==k)&(ps!=k))
        pr=tp/(tp+fp) if tp+fp else 0.;rc=tp/(tp+fn) if tp+fn else 0.
        precision.append(pr);recall.append(rc);f1.append(2*pr*rc/(pr+rc) if pr+rc else 0.)
    return dict(accuracy=float((ys==ps).mean()),macro_f1=float(np.mean(f1)),
                macro_precision=float(np.mean(precision)),macro_recall=float(np.mean(recall)),
                phase_accuracy=phase_correct/frames,loss=total_loss/frames)


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--data',default='synthetic.npz')
    a.add_argument('--make-synthetic',action='store_true')
    a.add_argument('--epochs',type=int,default=80)
    a.add_argument('--batch-size',type=int,default=32)
    a.add_argument('--lr',type=float,default=1e-4)
    a.add_argument('--num-errors',type=int,default=5)
    a.add_argument('--ablation',choices=['full','no_attention','no_gate','no_phase','pose_only'],default='full')
    a.add_argument('--out',default='runs/full')
    a.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a.add_argument('--seed',type=int,default=42)
    args=a.parse_args()
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    if args.make_synthetic: generate_synthetic(args.data)
    data=MotionDataset(args.data)
    splits=subject_split(data.subjects,args.seed)
    loaders=[DataLoader(MotionDataset(args.data,ids),batch_size=args.batch_size,
                        shuffle=(split==0),num_workers=0) for split,ids in enumerate(splits)]
    model=CMFErrNet(num_errors=args.num_errors,ablation=args.ablation).to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(.9,.999))
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
    outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True)
    best=-1.;history=[]
    for epoch in range(args.epochs):
        model.train();losses=[]
        for batch in loaders[0]:
            batch={k:v.to(args.device) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs=model(batch['pose'],batch['imu'])
            loss,_=compute_loss(outputs,batch['errors'],batch['phases'],batch.get('severity'))
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();losses.append(loss.item())
        schedule.step()
        val=metrics(model,loaders[1],args.device,args.num_errors)
        row={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation':val}
        history.append(row)
        print(f"epoch={epoch+1:03d} train_loss={row['train_loss']:.4f} val_acc={val['accuracy']:.4f}",flush=True)
        if val['macro_f1']>best:
            best=val['macro_f1']
            torch.save({'model':model.state_dict(),'num_errors':args.num_errors,
                        'ablation':args.ablation,'epoch':epoch+1},outdir/'best.pt')
        (outdir/'history.json').write_text(json.dumps(history,indent=2))
    ckpt=torch.load(outdir/'best.pt',map_location=args.device,weights_only=True)
    model.load_state_dict(ckpt['model'])
    test=metrics(model,loaders[2],args.device,args.num_errors)
    (outdir/'test_metrics.json').write_text(json.dumps(test,indent=2))
    print('HELD-OUT TEST:',json.dumps(test,indent=2))

if __name__=='__main__':main()
