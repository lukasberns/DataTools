from __future__ import print_function
import numpy as np
import torch 
import h5py
import matplotlib.pyplot as plt

def geometricPad(grid, pad=1):
  M = grid.shape[2]; # 0:batchsize, 1:channels, 2:width, 3:height
  M_new = pad+M+pad;
  new_shape = (grid.shape[0], grid.shape[1], M_new, M_new)
  grid_new = grid.new_zeros(new_shape);
  grid_new[:,:,pad:M+pad,pad:M+pad] = grid;
  grid_new[:,:,0:pad,pad:(M+pad)] = grid[:,:,:,0:pad].flip(-1).transpose(-1,-2);
  grid_new[:,:,pad:(M+pad),0:pad] = grid[:,:,0:pad,:].flip(-2).transpose(-1,-2);
  grid_new[:,:,(M+pad):(M+pad+pad),(pad):(M+pad)] = grid[:,:,:,(M-pad):].flip(-1).transpose(-1,-2);
  grid_new[:,:,pad:(M+pad),(M+pad):(M+pad+pad)] = grid[:,:,(M-pad):,:].flip(-2).transpose(-1,-2);
  return(grid_new);

# f = h5py.File('/home/lukasb/watchmal/data_ssd/IWCDmPMT_4pi_full_tank/h5_topo/gamma/IWCDmPMT_4pi_full_tank_gamma_E0to1000MeV_unif-pos-R371-y521cm_4pi-dir_3000evts_300.h5','r')
# event_data = f['event_data'][:]
# mGridCoords = np.stack([
#     f['mGridX'][()]/1000.,
#     f['mGridY'][()]/1000.,
#     f['mGridZ'][()]/1000.,
#     f['mGridDirX'][()]*3.,
#     f['mGridDirY'][()]*3.,
#     f['mGridDirZ'][()]*3.
# ],2).astype(np.float32)

def imshowRdBu(ax,img):
    minmax = max(torch.max(img),-torch.min(img))
    return ax.imshow(-img,cmap='RdBu',vmin=-minmax,vmax=minmax)

from torch.utils.data import Dataset, DataLoader

"""
Convert charge, time (each a 2D numpy array) to complex rep
"""
def np_QT2XY(evQ, evT):
    evR = np.power(evQ,0.25);
    evA = 2*np.pi*(evT-960.)/(1900.-600.)
    evX = evR*np.cos(evA);
    evY = evR*np.sin(evA);
    return evX, evY

"""
Convert charge, time (each a 3D pytorch tensor [B,H,W]) to complex rep
"""
def torch_QT2XY(evQ, evT):
    evR = torch.pow(evQ,0.25);
    evA = 2*np.pi*(evT-960.)/(1900.-600.)
    evX = evR*torch.cos(evA);
    evY = evR*torch.sin(evA);
    return evX, evY
    
"""
Inverse of np_QT2XY
"""
def np_XY2QT(evX, evY):
    evR = np.sqrt(evX*evX + evY*evY)
    evA = np.arctan2(evY, evX)
    evQ = np.power(evR,4);
    evT = 960. + evA*(1900.-600.)/(2.*np.pi)
    return evQ, evT

"""
Inverse of torch_QT2XY
"""
def torch_XY2QT(evX, evY):
    evR = torch.sqrt(evX*evX + evY*evY)
    evA = torch.atan2(evY, evX)
    evQ = torch.pow(evR,4);
    evT = 960. + evA*(1900.-600.)/(2.*np.pi)
    return evQ, evT

"""
Normalize time (each a 2D numpy array)
"""
def np_QT_timenorm(evQ, evT):
    evS = np.where(evQ > 0., 10.*(evT-960.)/(1900.-600.), 0.)
    return evQ, evS

def loadGridCoords(file_name):
    print("Reading PMT grid information from %s" % file_name)
    import h5py
    f = h5py.File(file_name,mode='r')
    mGridCoords = np.stack([
        f['mGridX'][()]/1000.,
        f['mGridY'][()]/1000.,
        f['mGridZ'][()]/1000.,
        f['mGridDirX'][()]*3.,
        f['mGridDirY'][()]*3.,
        f['mGridDirZ'][()]*3.
    ],2).astype(np.float32)
    print("mGridCoords.shape = ", mGridCoords.shape)
    f.close()
    return mGridCoords

class H5Dataset(Dataset):

    def __init__(self, files, transform=None, flavour=None, limit_num_files=0, start_fraction=0., use_fraction=1.0, QT_transform=np_QT2XY):
        """                                                                                                                                             
        Args: data_dirs ... a list of data directories to find files (up to 10 files read from each dir)                                                
              transform ... a function applied to pre-process data                                                                                      
              flavour ..... a string that is required to be present in the filename                                                                     
              limit_num_files ... an integer limiting number of files to be taken per data directory                                                    
              start_fraction ... a floating point fraction (0.0=>1.0) to specify which entry to start reading (per file)                                
              use_fraction ..... a floating point fraction (0.0=>1.0) to specify how much fraction of a file to be read out (per file)                  
              QT_transform   ... a function taking Q,T np arrays as arguments and transforming to the X,Y variables to use during training. If set to None, no transformation will be done.
        """
        self._transform = transform
        self._QT_transform = QT_transform
        self._files = []

        # Check input fractions makes sense                                                                                                             
        assert start_fraction >= 0. and start_fraction < 1.
        assert use_fraction > 0. and use_fraction <= 1.
        assert (start_fraction + use_fraction) <= 1.

        # Load files (up to 10) from each directory in data_dirs list                                                                                   
        # for d in data_dirs:
        #     file_list = [ os.path.join(d,f) for f in os.listdir(d) if flavour is None or flavour in f ]
        #     if limit_num_files: file_list = file_list[0:limit_num_files]
        #     self._files += file_list
        self._files = files

        self._file_handles = [None] * len(self._files)
        self._event_to_file_index  = []
        self._event_to_entry_index = []
        import h5py
        for file_index, file_name in enumerate(self._files):
            f = h5py.File(file_name,mode='r')
            data_size = f['event_data'].shape[0]
            start_entry = int(start_fraction * data_size)
            num_entries = int(use_fraction * data_size)
            self._event_to_file_index += [file_index] * num_entries
            self._event_to_entry_index += range(start_entry, start_entry+num_entries)
            f.close()

    def __len__(self):
        return len(self._event_to_file_index)

    def __getitem__(self,idx):
        file_index = self._event_to_file_index[idx]
        entry_index = self._event_to_entry_index[idx]
        if self._file_handles[file_index] is None:
            import h5py
            self._file_handles[file_index] = h5py.File(self._files[file_index],mode='r')
        fh = self._file_handles[file_index]
        label = fh['labels'][entry_index]
        #labelTranslation = [-1,0,1,-1,2] # make sure that labels 1,2,4 get values 0,1,2
        if label == 1: # electron
            label = 0
        elif label == 2: # muon
            label = 1
        elif label == 0: # gamma
            label = 2
        else:
            print('Unknown label', label, 'for entry_index', entry_index, 'treating as label=0')
            label = 0
        
        # try:
        #     label = labelTranslation[label]
        # except IndexError:
        #     print('IndexError at entry', entry_index, 'label in file was', fh['labels'][entry_index], 'translations are', labelTranslation)
        #     raise
        
        event_data = fh['event_data'][entry_index]
        # convert event data to complex rep
        if event_data.shape[2] == 2:
            evQ = event_data[:,:,0:1]
            evT = event_data[:,:,1:2]
        elif event_data.shape[2] == 38:
            evQ = event_data[:,:, 0:19]
            evT = event_data[:,:,19:38]
        else:
            raise Exception("Unexpected shape at index 2 (currently 2 and 38 are supported): (%d,%d,%d,..)" % event_data.shape)

        if self._QT_transform is not None:
            evX,evY = self._QT_transform(evQ, evT)
        else:
            evX,evY = evQ,evT
        totalQ = np.sum(evQ)
        
        p = np.sum(fh['directions'][entry_index,:,:] * np.expand_dims(fh['energies'][entry_index,:], 1), 0);
        totdir = p / np.expand_dims(np.sqrt(np.sum(p*p,0)),0)
        
        return np.concatenate([evX,evY],2),label,idx,fh['positions'][entry_index,0],totdir,np.sum(fh['energies'][entry_index,:]),totalQ
        # the ,0 in positions selects the pos for the first track (in case of gamma)
        # for directions we compute the energy-weighted sum (which is the photon momentum for a gamma)
        # thus positions and directions are just np arrays with 3 elements

def HKCollate(batch):
    data  = np.stack([sample[0] for sample in batch])
    label = [sample[1] for sample in batch]
    idx   = [sample[2] for sample in batch]
    pos   = np.stack([sample[3] for sample in batch])
    direc = np.stack([sample[4] for sample in batch])
    ene   = np.stack([sample[5] for sample in batch])
    totQ  = np.stack([sample[6] for sample in batch])
    return data,label,idx,pos,direc,ene,totQ

masses = np.array([0.511, 105.7, 0.511*2])
pthres = np.array([0.57,  118.,  0.57 *2])
Ethres = np.sqrt(np.power(masses,2) + np.power(pthres,2))

speedOfLight_m_s = 299792458.
speedOfLight_cm_ns = speedOfLight_m_s * 100. / 1e9

# fiTQun.parameters.dat:
# < fiTQun.WaterRefractiveIndex = 1.38 > Refractive index of water assumed by fitter
refractiveIndex_water = 1.38


def backward(blob):
    blob.optimizer.zero_grad()  # Reset gradients accumulation
    blob.loss.backward()
    blob.optimizer.step()

def save_state(blob):
    # Output file name
    filename = 'checkpoints/%s-%d.ckpt' % (blob.prefix, blob.iteration)
    # Save parameters
    # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
    # 2) network weight
    torch.save({
        'global_epoch': blob.epoch,
        'global_step': blob.iteration,
        'optimizer': blob.optimizer.state_dict(),
        'state_dict': blob.net.state_dict()
        }, filename)
    return filename

def restore_state(blob, iteration):
    # Open a file in read-binary mode
    weight_file = 'checkpoints/%s-%d.ckpt' % (blob.prefix, iteration)
    with open(weight_file, 'rb') as f:
        # torch interprets the file, then we can access using string keys
        checkpoint = torch.load(f)
        # load network weights
        blob.net.load_state_dict(checkpoint['state_dict'], strict=False)
        # if optimizer is provided, load the state of the optimizer
        if blob.optimizer is not None:
            blob.optimizer.load_state_dict(checkpoint['optimizer'])
        # load iteration count
        blob.epoch     = checkpoint['global_epoch']
        blob.iteration = checkpoint['global_step']

# weight_file = save_state(blob, '20190819-02-DeepTaylor-01-BatchNorm')
# print('Saved to', weight_file)


# # Recreate the network (i.e. initialize)
# blob.net=LeNet().cuda()
# # Get one batch of data to test
# blob.data, blob.label = next(iter(train_loader))
# # Run forward function
# res = forward(blob,True)
# # Report
# print('Accuracy:',res['accuracy'])



# # Restore the state
# restore_state(weight_file,blob)
# # Run the forward function
# res = forward(blob,True)
# # Report
# print('Accuracy',res['accuracy'])
