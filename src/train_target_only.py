# src/train_target_only.py
import os, copy, torch
from torch.utils.data import DataLoader, TensorDataset
from data_loader import get_data_and_scalers
from models import AlignHeteroMLP
from losses import heteroscedastic_nll, batch_r2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OPAMP_TYPE='5t_opamp'; BATCH=256; LR=1e-4; WD=1e-4; EPOCHS=80; PATIENCE=20; ALPHA_R2=1e-3

def dl(x,y,bs,shuffle,drop_last=True):
    x=torch.tensor(x,dtype=torch.float32); y=torch.tensor(y,dtype=torch.float32)
    return DataLoader(TensorDataset(x,y),batch_size=bs,shuffle=shuffle,drop_last=drop_last)

def main():
    data=get_data_and_scalers(opamp_type=OPAMP_TYPE)
    Xtr,Ytr=data['target_train']; Xva,Yva=data['target_val']
    m=AlignHeteroMLP(Xtr.shape[1],Ytr.shape[1],hidden_dim=512,num_layers=6,dropout_rate=0.1).to(DEVICE)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD)
    dtr=dl(Xtr,Ytr,BATCH,True,True); dva=dl(Xva,Yva,BATCH,False,False)
    best=float('inf'); best_state=None; pat=PATIENCE
    for ep in range(EPOCHS):
        m.train(); tr=0.0
        for xb,yb in dtr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            mu, logv, _ = m(xb)
            nll = heteroscedastic_nll(mu, logv, yb)
            r2  = (1.0 - batch_r2(yb, mu)).mean()
            loss = nll + ALPHA_R2*r2
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
            opt.step(); tr += loss.item()
        m.eval(); va=0.0
        with torch.no_grad():
            for xb,yb in dva:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                mu, logv, _ = m(xb)
                va += heteroscedastic_nll(mu, logv, yb).item()
        va /= max(1,len(dva))
        print(f"[Target-only] {ep+1}/{EPOCHS}  train {tr/len(dtr):.4f}  valNLL {va:.4f}")
        if va<best:
            best=va; best_state=copy.deepcopy(m.state_dict()); pat=PATIENCE
            os.makedirs('results',exist_ok=True)
            torch.save(best_state,f'results/{OPAMP_TYPE}_target_only_hetero.pth')
        else:
            pat-=1
            if pat==0: break
    print(f"已保存: ../results/{OPAMP_TYPE}_target_only_hetero.pth")

if __name__=='__main__': main()
