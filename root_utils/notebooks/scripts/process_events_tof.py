import os
import numpy as np

mcdir = '/home/lukasb/watchmal/data/IWCDmPMT_4pi_full_tank/h5_topo'
pname = 'e-'
#pname = 'mu-'
#pname = 'gamma'

outdir = '/home/lukasb/watchmal/data/IWCDmPMT_4pi_full_tank/reco_%s' % blob.prefix
if not os.path.isdir(outdir):
    os.mkdir(outdir)
if not os.path.isdir('%s/%s' % (outdir,pname)):
    os.mkdir('%s/%s' % (outdir,pname))

for bch in np.arange(99)+2:
    print("bch = %d" % bch)
    infile = '%s/%s/IWCDmPMT_4pi_full_tank_%s_E0to1000MeV_unif-pos-R371-y521cm_4pi-dir_3000evts_%d.h5' % (mcdir,pname,pname,bch)
    if not os.path.exists(infile):
        print("Skipping %s (not found)" % infile)
        continue

    files = [infile]
    
    dataset = H5Dataset(files,start_fraction=0.0,use_fraction=1.0);
    loader  = DataLoader(dataset,batch_size=32,shuffle=False,num_workers=4,collate_fn=HKCollate)

    blob.mGridCoords = loadGridCoords(files[0])

    simple_position, simple_logSigmaPosSqr, pred_Eabovethres, pred_logSigmaESqr, pred_position, pred_logSigmaPosSqr, pred_direction, pred_logSigmaDirSqr, label, positions, directions, energies = inferenceWithSoftmax(blob,loader)

    outfile = '%s/%s/IWCDmPMT_4pi_full_tank_%s_E0to1000MeV_unif-pos-R371-y521cm_4pi-dir_3000evts_%d.h5' % (outdir,pname,pname,bch)
    of = h5py.File(outfile, "w")

    def writeDataset(of, x, name, dtype):
        dataset = of.create_dataset(name, x.shape, dtype=dtype)
        dataset[...] = x

    # simple pred info (just used for tof subtraction)
    writeDataset(of, simple_position, 'pred_simple_position', 'f')                  # (N,3)
    writeDataset(of, simple_logSigmaPosSqr[:,0], 'pred_simple_logSigmaPosSqr', 'f') # (N,)

    # pred info
    writeDataset(of, pred_Eabovethres[:,0], 'pred_Eabovethres', 'f')       # (N,)
    writeDataset(of, pred_logSigmaESqr[:,0], 'pred_logSigmaESqr', 'f')     # (N,)
    writeDataset(of, pred_position, 'pred_position', 'f')                  # (N,3)
    writeDataset(of, pred_logSigmaPosSqr[:,0], 'pred_logSigmaPosSqr', 'f') # (N,)
    writeDataset(of, pred_direction, 'pred_direction', 'f')                # (N,3)
    writeDataset(of, pred_logSigmaDirSqr[:,0], 'pred_logSigmaDirSqr', 'f') # (N,)

    # true info
    writeDataset(of, np.arange(label.shape[0])+1, 'nevt', 'i') # (N,) this is the event number, to be matched with nevt in fiTQun
    writeDataset(of, label, 'true_label', 'i')           # (N,)
    writeDataset(of, positions, 'true_positions', 'f')   # (N,3)
    writeDataset(of, directions, 'true_directions', 'f') # (N,3)
    writeDataset(of, energies, 'true_energies', 'f')     # (N,)
    writeDataset(of, energies - Ethres[label], 'true_Eabovethres', 'f') # (N,)

    of.close()
    print("Wrote to %s" % outfile)

print("done")
