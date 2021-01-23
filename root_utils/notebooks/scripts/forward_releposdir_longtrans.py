
def forward(blob,train=True):
    """
       Args: blob should have attributes, net, criterion, softmax, data, label
       Returns: a dictionary of predicted labels, softmax, loss, and accuracy
    """
    with torch.set_grad_enabled(train):
        # Prediction
        data = torch.as_tensor(np.concatenate([blob.data, np.broadcast_to(blob.mGridCoords, (blob.data.shape[0],)+blob.mGridCoords.shape)],3)).cuda()#[torch.as_tensor(d).cuda() for d in blob.data]
        data = data.permute(0,3,1,2)
        out = blob.net(data)
        out_Eabovethres,out_Eres = getEnergyPrediction(out)
        out_pos,out_posres       = getPositionPrediction(out)
        out_dir,out_dirres       = getDirectionPrediction(out)

        true_dir = torch.as_tensor(blob.direction).type(torch.FloatTensor).cuda()
        
        # Training
        nll_energy = getNllEnergy(
            out_Eabovethres,
            out_Eres,
            torch.as_tensor(getTrueEnergyAboveThreshold(blob)).type(torch.FloatTensor).unsqueeze(1).cuda()
        )
        nll_position = getNllPosition(
            out_pos,
            out_posres,
            torch.as_tensor(blob.position/1000.).type(torch.FloatTensor).cuda(),
            true_dir
        )
        nll_direction = getNllDirection(
            out_dir,
            out_dirres,
            true_dir
        )
        #print('nll_energy.shape', nll_energy.shape, torch.mean(nll_energy), torch.std(nll_energy))
        #print('nll_position.shape', nll_position.shape, torch.mean(nll_position), torch.std(nll_position))
        loss_energy = torch.sum(nll_energy)
        loss_position = torch.sum(nll_position)
        loss_direction = torch.sum(nll_direction)
        loss = loss_energy + loss_position + loss_direction
        blob.loss = loss
        
        batchSize = float(out.shape[0])
        mean_chi2_E   = 2.*loss_energy  .cpu().detach().item() / batchSize
        mean_chi2_pos = 2.*loss_position.cpu().detach().item() / batchSize
        mean_chi2_dir = 2.*loss_direction.cpu().detach().item() / batchSize
        mean_Eres   = torch.exp(torch.mean(out_Eres  )).cpu().detach().item()
        mean_posres = torch.exp(torch.mean(out_posres)).cpu().detach().item()
        mean_dirres = torch.exp(torch.mean(out_dirres)).cpu().detach().item()
        
        return {'pred_Eabovethres' : out_Eabovethres.cpu().detach().numpy(),
                'pred_Eres'        : out_Eres       .cpu().detach().numpy(),
                'pred_position'    : out_pos        .cpu().detach().numpy(),
                'pred_positionres' : out_posres     .cpu().detach().numpy(),
                'pred_direction'    : out_dir        .cpu().detach().numpy(),
                'pred_directionres' : out_dirres     .cpu().detach().numpy(),
                'loss_energy'   : loss_energy  .cpu().detach().item(),
                'loss_position' : loss_position.cpu().detach().item(),
                'loss_direction': loss_direction.cpu().detach().item(),
                'loss'         : loss         .cpu().detach().item(),
                'mean_chi2_E'   : mean_chi2_E,
                'mean_chi2_pos' : mean_chi2_pos,
                'mean_chi2_dir' : mean_chi2_dir,
                'mean_Eres'     : mean_Eres,
                'mean_posres'   : mean_posres,
                'mean_dirres'   : mean_dirres,
               }

resKeysToLog = ['loss_energy','loss_position','loss_direction','loss','mean_chi2_E','mean_chi2_pos','mean_chi2_dir','mean_Eres','mean_posres','mean_dirres']
