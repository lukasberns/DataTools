
def inferenceWithSoftmax(blob,data_loader,use_relE=True):
    label,pred_pid_index,pred_pid_softmax,pred_Eabovethres,pred_logSigmaESqr,pred_position,pred_logSigmaPosSqr,pred_direction,pred_logSigmaDirSqr,positions,directions,energies=[],[],[],[],[],[],[],[],[],[],[],[]
    # set the network to test (non-train) mode
    blob.net.eval()
    # create the result holder
    index,label,prediction = [],[],[]
    for i,data in enumerate(data_loader):
        blob.data, blob.label = data[0:2]
        blob.position  = data[3]
        blob.direction = data[4]
        blob.energy    = data[5]
        blob.totQ      = data[6]
        totQ           = data[6]
        res = forward(blob,train=False)
        
        Evis = np.expand_dims(0.085*blob.totQ, -1)
        pred_pid_index  .append(res['pred_pid_index'])
        pred_pid_softmax.append(res['pred_pid_softmax'])
        if use_relE:
            pred_Eabovethres.append(res['pred_Eabovethres']*(np.sqrt(500.)*np.sqrt(Evis+0.5)) + Evis)
        else:
            pred_Eabovethres.append(res['pred_Eabovethres'])
        pred_logSigmaESqr.append(res['pred_Eres'])
        pred_position.append(res['pred_position'])
        pred_logSigmaPosSqr.append(res['pred_positionres'])
        pred_direction.append(res['pred_direction'])
        pred_logSigmaDirSqr.append(res['pred_directionres'])
        
        label.append(blob.label)
        positions.append(blob.position)
        directions.append(blob.direction)
        energies.append(blob.energy)
        #if i==2: break
    # report accuracy
    pred_pid_index        = np.hstack(pred_pid_index)
    pred_pid_softmax      = np.vstack(pred_pid_softmax)
    pred_Eabovethres      = np.vstack(pred_Eabovethres)
    pred_logSigmaESqr     = np.vstack(pred_logSigmaESqr)
    pred_position         = np.vstack(pred_position)
    pred_logSigmaPosSqr   = np.vstack(pred_logSigmaPosSqr)
    pred_direction        = np.vstack(pred_direction)
    pred_logSigmaDirSqr   = np.vstack(pred_logSigmaDirSqr)
    label      = np.hstack(label)
    positions  = np.vstack(positions)
    directions = np.vstack(directions)
    energies   = np.concatenate(energies)
    
    return pred_pid_index, pred_pid_softmax, pred_Eabovethres, pred_logSigmaESqr, pred_position, pred_logSigmaPosSqr, pred_direction, pred_logSigmaDirSqr, label, positions, directions, energies
