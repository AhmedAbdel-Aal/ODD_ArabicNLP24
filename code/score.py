def msp_scores(id_logits, ood_logits):
    id_scores = np.max(torch.softmax(id_logits, dim=1).detach().cpu().numpy(), axis=1)
    ood_scores = np.max(torch.softmax(ood_logits, dim=1).detach().cpu().numpy(), axis=1)
    return id_scores, ood_scores


def energy_scores(id_logits, ood_logits, T = 1.0):
    # calculate Energy scores
    id_scores = torch.logsumexp(id_logits.data.cpu(), dim=1).numpy()
    ood_scores = torch.logsumexp(ood_logits.data.cpu(), dim=1).numpy()
    
def conformal_score(id_logits, ood_logits, id_labels):
    id_smx = torch.softmax(id_logits, dim=1).detach().cpu().numpy()
    
    n=1000
    alpha = 0.1
    # Split the softmax scores into calibration and validation sets (save the shuffling)
    idx = np.array([1] * n + [0] * (id_smx.shape[0]-n)) > 0
    np.random.shuffle(idx)
    cal_smx, val_smx = id_smx[idx,:], id_smx[~idx,:]
    cal_labels, val_labels = id_labels[idx], id_labels[~idx]
    
    # 1: get conformal scores. n = calib_Y.shape[0]
    cal_scores = 1-cal_smx[np.arange(n),cal_labels]
    
    # 2: get adjusted quantile
    q_level = np.ceil((n+1)*(1-alpha))/n
    qhat = np.quantile(cal_scores, q_level, interpolation='higher')
    
    prediction_sets = val_smx >= (1-qhat) # 3: form prediction sets
    
 
    ood_smx = torch.softmax(ood_logits, dim=1).detach().cpu().numpy()

    id_nc_scores = np.max((val_smx - (1-qhat)), axis=1)
    ood_nc_scores = np.max((ood_smx - (1-qhat)), axis=1)

    return id_nc_scores, ood_nc_scores

