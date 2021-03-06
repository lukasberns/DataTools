from __future__ import print_function
import os
import numpy as np
import sys

try: mcdir
except NameError: mcdir = '/home/lukasb/watchmal/data_ssd/IWCDmPMT_4pi_full_tank/h5_topo'

try: batches
except NameError: batches = np.arange(300)+100

try: pnameset
except NameError: pnameset = "all"

if pnameset == "all":
    pnames = ('e-','mu-','gamma')
elif pnameset == "e-mu-":
    pnames = ('e-','mu-',)
elif pnameset == "electron":
    pnames = ('e-',)
elif pnameset == "muon":
    pnames = ('mu-',)
else:
    print("Unknown pnameset = '%s'" % pnameset)
    sys.exit(15)

try: filespattern
except NameError: filespattern = '%(mcdir)s/%(pname)s/IWCDmPMT_4pi_full_tank_%(pname)s_E0to1000MeV_unif-pos-R371-y521cm_4pi-dir_3000evts_%(bch)d.h5'

files = [filespattern % {"mcdir":mcdir,"pname":pname,"bch":bch} for bch in batches for pname in pnames]

try: QT_transform
except NameError: QT_transform = np_QT2XY

train_ds = H5Dataset(files,QT_transform=QT_transform,start_fraction=0.0,use_fraction=0.9);
test_ds  = H5Dataset(files,QT_transform=QT_transform,start_fraction=0.9,use_fraction=0.1);

blob.mGridCoords = loadGridCoords(files[0])

# for training
train_loader=DataLoader(train_ds,batch_size=32,shuffle=True,num_workers=4,collate_fn=HKCollate)
# for validation
test_loader =DataLoader( test_ds,batch_size=64,shuffle=True,num_workers=2,collate_fn=HKCollate)


import time
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

from utils import CSVData

def progress_text(count, total, message=''):
    toolbar_width = 50
    done_width = int(count/total * toolbar_width)
    if done_width > toolbar_width:
        done_width = toolbar_width
    sys.stdout.write("[%s%s] %s\r" % ("=" * done_width, " " * (toolbar_width-done_width), message))
    sys.stdout.flush()

def progress_done():
    sys.stdout.write("\n")
    sys.stdout.flush()

# Define train period. "epoch" = N image consumption where N is the total number of train samples.
try: TRAIN_EPOCH
except NameError: TRAIN_EPOCH = 16.0

log_suffix = "_epoch%.0fto%.0f" % (blob.epoch, TRAIN_EPOCH)
print("Logging to %s-log_{train,test}%s.csv" % (blob.prefix,log_suffix))
blob.train_log, blob.test_log = CSVData('%s-log_train%s.csv' % (blob.prefix,log_suffix)), CSVData('%s-log_test%s.csv' % (blob.prefix,log_suffix))

# Set the network to training mode
blob.net.train()

def getMessage(blob, res):
    if 'loss_pid' in res:
        return '... Iteration %d ... Epoch %1.2f ... Loss %1.3f = %1.3f (pid) + %1.3f (energy) + %1.3f (pos) + %1.3f (dir)' % (blob.iteration,blob.epoch,res['loss'],res['loss_pid'],res['loss_energy'],res['loss_position'],res['loss_direction'])
    else:
        return '... Iteration %d ... Epoch %1.2f ... Loss %1.3f = %1.3f (energy) + %1.3f (pos) + %1.3f (dir)' % (blob.iteration,blob.epoch,res['loss'],res['loss_energy'],res['loss_position'],res['loss_direction'])

# Start training
while int(blob.epoch+0.5) < TRAIN_EPOCH:
    print('Epoch',int(blob.epoch+0.5),'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # Create a progress bar for this epoch
    #progress = display(progress_bar(0,len(train_loader)),display_id=True)
    progress_text(0,len(train_loader))
    # Loop over data samples and into the network forward function
    for i,data in enumerate(train_loader):
        # Data and label
        # data,label,idx,pos,direc,ene
        blob.data,blob.label = data[0:2]
        blob.position = data[3]
        blob.direction= data[4]
        blob.energy   = data[5]
        blob.totQ     = data[6]
        # Call forward: make a prediction & measure the average error
        res = forward(blob,True)
        # Call backward: backpropagate error and update weights
        backward(blob)
        # Epoch update
        blob.epoch += 1./len(train_loader)
        blob.iteration += 1
        
        #
        # Log/Report
        #
        # Record the current performance on train set
        blob.train_log.record(['iteration','epoch']+resKeysToLog,[blob.iteration,blob.epoch]+[res[key] for key in resKeysToLog])
        blob.train_log.write()
        # once in a while, report
        if i==0 or (i+1)%10 == 0:
            message = getMessage(blob,res)
            #progress.update(progress_bar((i+1),len(train_loader),message))
            progress_text(i+1,len(train_loader),message)
        # more rarely, run validation
        if (i+1)%40 == 0:
            with torch.no_grad():
                blob.net.eval()
                test_data = next(iter(test_loader))
                blob.data,blob.label = test_data[0:2]
                blob.position = test_data[3]
                blob.direction= test_data[4]
                blob.energy   = test_data[5]
                blob.totQ     = test_data[6]
                res = forward(blob,False)
                blob.test_log.record(['iteration','epoch']+resKeysToLog,[blob.iteration,blob.epoch]+[res[key] for key in resKeysToLog])
                blob.test_log.write()
            blob.net.train()
        # even more rarely, save state
        if (i+1)%2000 == 0:
            with torch.no_grad():
                blob.net.eval()
                save_state(blob)
            blob.net.train()
        if blob.epoch >= TRAIN_EPOCH:
            break
    message = getMessage(blob,res)
    #print(message)
    #progress.update(progress_bar((i+1),len(train_loader),message))
    progress_text(i+1,len(train_loader),message)
    progress_done()

blob.test_log.close()
blob.train_log.close()

print("Done")

