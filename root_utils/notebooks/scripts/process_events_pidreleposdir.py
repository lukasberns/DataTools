import os
import numpy as np

try: mcdir
except NameError: mcdir = '/home/lukasb/watchmal/data/IWCDmPMT_4pi_full_tank/h5_topo'

try: pname
except NameError: pname = 'e-'

try: use_relE
except NameError: use_relE = True

#pname = 'e-'
#pname = 'mu-'
#pname = 'gamma'

try: filespattern
except NameError: filespattern = '%(mcdir)s/%(pname)s/IWCDmPMT_4pi_full_tank_%(pname)s_E0to1000MeV_unif-pos-R371-y521cm_4pi-dir_3000evts_%(bch)d.h5'

try: outdirpattern
except NameError: outdirpattern = '/home/lukasb/watchmal/data/IWCDmPMT_4pi_full_tank/reco_%s'

try: outfilepattern
except NameError: outfilepattern = '%(outdir)s/%(pname)s/IWCDmPMT_4pi_full_tank_%(pname)s_E0to1000MeV_unif-pos-R371-y521cm_4pi-dir_3000evts_%(bch)d.h5'

outdir = outdirpattern % blob.prefix

if not os.path.isdir(outdir):
    os.mkdir(outdir)
if not os.path.isdir('%s/%s' % (outdir,pname)):
    os.mkdir('%s/%s' % (outdir,pname))

for bch in np.arange(99)+2:
    print("bch = %d" % bch)
    infile = filespattern % {"mcdir":mcdir,"pname":pname,"bch":bch}
    if not os.path.exists(infile):
        print("Skipping %s (not found)" % infile)
        continue

    files = [infile]
    
    try: QT_transform
    except NameError: QT_transform = np_QT2XY

    dataset = H5Dataset(files,QT_transform=QT_transform,start_fraction=0.0,use_fraction=1.0);
    loader  = DataLoader(dataset,batch_size=32,shuffle=False,num_workers=4,collate_fn=HKCollate)

    blob.mGridCoords = loadGridCoords(files[0])

    pred_pid_index, pred_pid_softmax, pred_Eabovethres, pred_logSigmaESqr, pred_position, pred_logSigmaPosSqr, pred_direction, pred_logSigmaDirSqr, label, positions, directions, energies = inferenceWithSoftmax(blob,loader,use_relE=use_relE)

    outfile = outfilepattern % {"outdir":outdir,"pname":pname,"bch":bch}
    of = h5py.File(outfile, "w")

    def writeDataset(of, x, name, dtype):
        dataset = of.create_dataset(name, x.shape, dtype=dtype)
        dataset[...] = x

    # pred info
    writeDataset(of, pred_pid_index, 'pred_pid_index', 'i')                # (N,)
    writeDataset(of, pred_pid_softmax, 'pred_pid_softmax', 'f')            # (N,)
    if pred_Eabovethres.shape[1] == 1:
        writeDataset(of, pred_Eabovethres[:,0], 'pred_Eabovethres', 'f')       # (N,)
        writeDataset(of, pred_logSigmaESqr[:,0], 'pred_logSigmaESqr', 'f')     # (N,)
    else:
        writeDataset(of, pred_Eabovethres, 'pred_Eabovethres', 'f')       # (N,*)
        writeDataset(of, pred_logSigmaESqr, 'pred_logSigmaESqr', 'f')     # (N,*)
    writeDataset(of, pred_position, 'pred_position', 'f')                  # (N,3)
    if pred_logSigmaPosSqr.shape[1] == 1:
        writeDataset(of, pred_logSigmaPosSqr[:,0], 'pred_logSigmaPosSqr', 'f') # (N,)
    else:
        writeDataset(of, pred_logSigmaPosSqr, 'pred_logSigmaPosSqr', 'f') # (N,*), for longtrans etc.
    writeDataset(of, pred_direction, 'pred_direction', 'f')                # (N,3)
    if pred_logSigmaDirSqr.shape[1] == 1:
        writeDataset(of, pred_logSigmaDirSqr[:,0], 'pred_logSigmaDirSqr', 'f') # (N,)
    else:
        writeDataset(of, pred_logSigmaDirSqr, 'pred_logSigmaDirSqr', 'f') # (N,*)

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
