"""
Python 3 script for processing a list of ROOT files into .npz files

To keep references to the original ROOT files, the file path is stored in the output.

An index is saved for every event in the output npz file corresponding to the event index within that ROOT file (ev).

Authors: Wojtek Fedorko, Julian Ding, Nick Prouse
"""

import argparse
from root_utils.root_file_utils import *
from root_utils.pos_utils import *

ROOT.gROOT.SetBatch(True)

def get_args():
    parser = argparse.ArgumentParser(description='dump WCSim data into numpy .npz file')
    parser.add_argument('input_files', type=str, nargs='+')
    parser.add_argument('-d', '--output_dir', type=str, default=None)
    args = parser.parse_args()
    return args


def dump_file(infile, outfile):
    label = get_label(infile)

    wcsim = WCSimFile(infile)

    # All data arrays are initialized here
    ev_ids = []
    ev_data = []
    labels = []
    pids = []
    positions = []
    directions = []
    energies = []
    files = []

    for ev in range(wcsim.nevent):
        # if ev%100 == 0:
        #    print("now processing event " +str(ev))

        wcsim.get_event(ev)

        # if ev%100 == 0:
        #    print("number of sub events: " + str(wcsim.ntrigger))
        direction, energy, pid, position = wcsim.get_truth_info()
        if len(energy) < 1:
            print("event", ev, "has no initial particle tracks in WCSim output - can't find truth info!")
            continue

        trigger = wcsim.get_first_trigger()

        ncherenkovdigihits = trigger.GetNcherenkovdigihits()

        if ncherenkovdigihits == 0:
            print("event, trigger has no hits " + str(ev) + " " + str(wcsim.current_trigger))
            continue

        np_q = np.zeros(ncherenkovdigihits)
        np_t = np.zeros(ncherenkovdigihits)

        np_pmt_index = np.zeros(ncherenkovdigihits, dtype=np.int32)

        for i in range(ncherenkovdigihits):
            wcsimrootcherenkovdigihit = trigger.GetCherenkovDigiHits().At(i)

            hit_q = wcsimrootcherenkovdigihit.GetQ()
            hit_t = wcsimrootcherenkovdigihit.GetT()
            hit_tube_id = wcsimrootcherenkovdigihit.GetTubeId() - 1

            np_pmt_index[i] = hit_tube_id
            np_q[i] = hit_q
            np_t[i] = hit_t

        np_module_index = module_index(np_pmt_index)
        np_pmt_in_module_id = pmt_in_module_id(np_pmt_index)

        np_wall_indices = np.where(is_barrel(np_module_index))

        np_q_wall = np_q[np_wall_indices]
        np_t_wall = np_t[np_wall_indices]

        np_wall_row, np_wall_col = row_col(np_module_index[np_wall_indices])
        np_pmt_in_module_id_wall = np_pmt_in_module_id[np_wall_indices]

        np_wall_data_rect = np.zeros((16, 40, 38))
        np_wall_data_rect[np_wall_row,
                          np_wall_col,
                          np_pmt_in_module_id_wall] = np_q_wall
        np_wall_data_rect[np_wall_row,
                          np_wall_col,
                          np_pmt_in_module_id_wall + 19] = np_t_wall

        np_wall_data_rect_ev = np.expand_dims(np_wall_data_rect, axis=0)

        # This part updates the data arrays
        ev_data.append(np_wall_data_rect_ev)
        labels.append(label)
        pids.append(pid)
        positions.append(position)
        directions.append(direction)
        energies.append(energy)

        ev_ids.append(ev)
        files.append(infile)

    # Readying all data arrays for saving
    all_events = np.concatenate(ev_data)
    all_labels = np.asarray(labels)
    all_pids = np.asarray(pids)
    all_positions = np.asarray(positions)
    all_directions = np.asarray(directions)
    all_energies = np.asarray(energies)
    all_ids = np.asarray(ev_ids)
    all_files = np.asarray(files, dtype=object)
    np.savez_compressed(outfile, event_data=all_events, labels=all_labels, pids=all_pids, positions=all_positions,
                        directions=all_directions, energies=all_energies, event_ids=all_ids, root_files=all_files)
    del wcsim


if __name__ == '__main__':

    config = get_args()
    if config.output_dir is not None:
        print("output directory: " + str(config.output_dir))
        if not os.path.exists(config.output_dir):
            print("                  (does not exist... creating new directory)")
            os.mkdir(config.output_dir)
        if not os.path.isdir(config.output_dir):
            raise argparse.ArgumentTypeError("Cannot access or create output directory" + config.output_dir)
    else:
        print("output directory not provided... output files will be in same locations as input files")

    file_count = len(config.input_files)
    current_file = 0

    for input_file in config.input_files:
        if os.path.splitext(input_file)[1].lower() != '.root':
            print("File " + input_file + " is not a .root file, skipping")
            continue
        input_file = os.path.abspath(input_file)

        if config.output_dir is None:
            output_file = os.path.splitext(input_file)[0] + '.npz'
        else:
            output_file = os.path.join(config.output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.npz')

        print("\nNow processing " + input_file)
        print("Outputting to " + output_file)

        dump_file(input_file, output_file)

        current_file += 1
        print("Finished converting file " + output_file + " (" + str(current_file) + "/" + str(file_count) + ")")

    print("\n=========== ALL FILES CONVERTED ===========\n")
