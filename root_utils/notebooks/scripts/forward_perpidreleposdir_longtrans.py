
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
        out_pid                  = getPIDPrediction(out)
        out_Eabovethres,out_Eres = getEnergyPrediction(out)
        out_pos,out_posres       = getPositionPrediction(out)
        out_dir,out_dirres       = getDirectionPrediction(out)

        true_dir = torch.as_tensor(blob.direction).type(torch.FloatTensor).cuda()
        
        # Training
        true_pid = torch.as_tensor(getTruePID(blob)).type(torch.LongTensor).cuda()
        true_pid.requires_grad = False
        nll_pid = getNllPid(
            out_pid,
            true_pid,
        )
        nll_energy = getNllEnergy(
            out_Eabovethres,
            out_Eres,
            torch.as_tensor(getTrueEnergyAboveThreshold(blob)).type(torch.FloatTensor).unsqueeze(1).cuda(),
            true_pid
        )
        nll_position = getNllPosition(
            out_pos,
            out_posres,
            torch.as_tensor(blob.position/1000.).type(torch.FloatTensor).cuda(),
            true_dir,
            true_pid
        )
        nll_direction = getNllDirection(
            out_dir,
            out_dirres,
            true_dir,
            true_pid
        )
        #print('nll_energy.shape', nll_energy.shape, torch.mean(nll_energy), torch.std(nll_energy))
        #print('nll_position.shape', nll_position.shape, torch.mean(nll_position), torch.std(nll_position))
        loss_pid    = torch.sum(nll_pid)
        loss_energy = torch.sum(nll_energy)
        loss_position = torch.sum(nll_position)
        loss_direction = torch.sum(nll_direction)
        loss = loss_pid + loss_energy + loss_position + loss_direction
        blob.loss = loss
        
        batchSize = float(out.shape[0])
        out_pid_softmax    = torch.nn.functional.softmax(out_pid,dim=1)
        out_pid_prediction = torch.argmax(out_pid,dim=-1)
        mean_pid_accuracy = (out_pid_prediction == true_pid).sum().item() / float(out_pid_prediction.nelement())
        mean_chi2_E   = 2.*loss_energy  .cpu().detach().item() / batchSize
        mean_chi2_pos = 2.*loss_position.cpu().detach().item() / batchSize
        mean_chi2_dir = 2.*loss_direction.cpu().detach().item() / batchSize
        mean_Eres   = torch.exp(torch.mean(out_Eres  )).cpu().detach().item()
        mean_posres = torch.exp(torch.mean(out_posres)).cpu().detach().item()
        mean_dirres = torch.exp(torch.mean(out_dirres)).cpu().detach().item()
        
        return {
                'pred_pid_index'   : out_pid_prediction.cpu().detach().numpy(),
                'pred_pid_softmax' : out_pid_softmax.cpu().detach().numpy(),
                'pred_Eabovethres' : out_Eabovethres.cpu().detach().numpy(),
                'pred_Eres'        : out_Eres       .cpu().detach().numpy(),
                'pred_position'    : out_pos        .cpu().detach().numpy(),
                'pred_positionres' : out_posres     .cpu().detach().numpy(),
                'pred_direction'    : out_dir        .cpu().detach().numpy(),
                'pred_directionres' : out_dirres     .cpu().detach().numpy(),
                'loss_pid'      : loss_pid      .cpu().detach().item(),
                'loss_energy'   : loss_energy   .cpu().detach().item(),
                'loss_position' : loss_position .cpu().detach().item(),
                'loss_direction': loss_direction.cpu().detach().item(),
                'loss'          : loss          .cpu().detach().item(),
                'mean_pid_accuracy' : mean_pid_accuracy,
                'mean_chi2_E'   : mean_chi2_E,
                'mean_chi2_pos' : mean_chi2_pos,
                'mean_chi2_dir' : mean_chi2_dir,
                'mean_Eres'     : mean_Eres,
                'mean_posres'   : mean_posres,
                'mean_dirres'   : mean_dirres,
               }

resKeysToLog = ['loss_pid','loss_energy','loss_position','loss_direction','loss','mean_pid_accuracy','mean_chi2_E','mean_chi2_pos','mean_chi2_dir','mean_Eres','mean_posres','mean_dirres']
