import glob
import logging
import os

import awkward as ak
import numpy as np
import pandas as pd
import uproot

# Constants for directories
ROOTFILE_DIR = "./data/4leptons"
SAVE_DIR = "./data/4leptons_converted"

# Configure logging globally
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def ensure_save_dir():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        logging.info(f"Created save directory: {SAVE_DIR}")


def load_root_files(rootfile_dir):
    root_files = glob.glob(f"{rootfile_dir}/*.root")
    logging.info(f"Found {len(root_files)} ROOT files in {rootfile_dir}")
    trees = []
    for root_file in root_files:
        with uproot.open(root_file) as file:
            analysis1 = file.get("analysis;1")
            keys = file.keys()
            if analysis1 is None:
                size_bytes = os.path.getsize(root_file)
                logging.warning(
                    f"No 'analysis;1' found in {os.path.basename(root_file)}, keys: {keys}, size: {size_bytes} bytes"
                )
                continue
            else:
                trees.append(analysis1)
                logging.info(f"Loaded tree from {os.path.basename(root_file)}")
    return trees


def convert_tree_to_particle_table(tree):
    lep_cols = ["lep_type", "lep_pt", "lep_eta", "lep_phi", "lep_e", "lep_charge"]
    tau_cols = ["tau_pt", "tau_eta", "tau_phi", "tau_e", "tau_charge"]
    photon_cols = ["photon_pt", "photon_eta", "photon_phi", "photon_e"]

    columns = ["eventNumber"] + lep_cols + tau_cols + photon_cols
    rows_out = []

    for arrays in uproot.iterate(tree, columns, step_size="100 MB", library="ak"):
        eventNumber = arrays["eventNumber"]

        # Leptons
        event_lep_broadcast, _ = ak.broadcast_arrays(eventNumber, arrays["lep_type"])
        lep_rows = np.column_stack(
            [
                ak.to_numpy(ak.flatten(event_lep_broadcast)),  # eventNumber
                ak.to_numpy(ak.flatten(arrays["lep_type"])),  # par_type
                ak.to_numpy(ak.flatten(arrays["lep_pt"])),  # particle_pt
                ak.to_numpy(ak.flatten(arrays["lep_eta"])),  # particle_eta
                ak.to_numpy(ak.flatten(arrays["lep_phi"])),  # particle_phi
                ak.to_numpy(ak.flatten(arrays["lep_e"])),  # particle_e
                ak.to_numpy(ak.flatten(arrays["lep_charge"])),  # particle_charge
            ]
        )

        # Taus
        event_tau_broadcasted, tau_jagged = ak.broadcast_arrays(
            eventNumber, arrays["tau_pt"]
        )
        event_flat = ak.flatten(event_tau_broadcasted)
        tau_pt_flat = ak.flatten(tau_jagged)
        tau_eta_flat = ak.flatten(arrays["tau_eta"])
        tau_phi_flat = ak.flatten(arrays["tau_phi"])
        tau_e_flat = ak.flatten(arrays["tau_e"])
        tau_charge_flat = ak.flatten(arrays["tau_charge"])
        tau_rows = np.column_stack(
            [
                ak.to_numpy(event_flat),  # eventNumber
                np.full(len(tau_pt_flat), 15),  # particle_type = 15
                ak.to_numpy(tau_pt_flat),  # particle_pt
                ak.to_numpy(tau_eta_flat),  # particle_eta
                ak.to_numpy(tau_phi_flat),  # particle_phi
                ak.to_numpy(tau_e_flat),  # particle_e
                ak.to_numpy(tau_charge_flat),  # particle_charge
            ]
        )

        # Photons
        event_photon_broadcasted, photon_jagged = ak.broadcast_arrays(
            eventNumber, arrays["photon_pt"]
        )
        event_flat = ak.flatten(event_photon_broadcasted)
        pho_pt_flat = ak.flatten(photon_jagged)
        pho_eta_flat = ak.flatten(arrays["photon_eta"])
        pho_phi_flat = ak.flatten(arrays["photon_phi"])
        pho_e_flat = ak.flatten(arrays["photon_e"])
        photon_rows = np.column_stack(
            [
                ak.to_numpy(event_flat),  # eventNumber
                np.full(len(pho_pt_flat), 22),  # particle_type = 22
                ak.to_numpy(pho_pt_flat),  # particle_pt
                ak.to_numpy(pho_eta_flat),  # particle_eta
                ak.to_numpy(pho_phi_flat),  # particle_phi
                ak.to_numpy(pho_e_flat),  # particle_e
                np.zeros(len(pho_pt_flat)),  # particle_charge = 0
            ]
        )

        chunk = np.vstack([lep_rows, tau_rows, photon_rows])
        rows_out.append(chunk)

    particle_table = np.vstack(rows_out)

    # Sort by eventNumber (column 0)
    sorted_indices = np.argsort(particle_table[:, 0])
    particle_table = particle_table[sorted_indices]

    logging.info(f"Converted tree to particle table with shape {particle_table.shape}")
    return particle_table


def converter(trees):
    all_particles = []
    for tree in trees:
        particle_table = convert_tree_to_particle_table(tree)
        all_particles.append(particle_table)

    all_particles_array = np.vstack(all_particles)
    logging.info(f"Combined all particle tables shape: {all_particles_array.shape}")

    columns = [
        "event_number",
        "particle_type",
        "particle_pt",
        "particle_eta",
        "particle_phi",
        "particle_e",
        "particle_charge",
    ]
    df = pd.DataFrame(all_particles_array, columns=columns)

    return all_particles_array, df


def save_npz(particle_table, filename="particle_table.npz"):
    ensure_save_dir()
    filepath = os.path.join(SAVE_DIR, filename)
    np.savez_compressed(filepath, particle_table=particle_table)
    logging.info(f"Saved particle_table as npz to {filepath}")


def save_dataframe_pickle(df, filename="particles_df.pkl"):
    ensure_save_dir()
    filepath = os.path.join(SAVE_DIR, filename)
    df.to_pickle(filepath)
    logging.info(f"Saved DataFrame as pickle to {filepath}")


if __name__ == "__main__":
    trees = load_root_files(ROOTFILE_DIR)
    particle_table, df = converter(trees)
    save_npz(particle_table)
    save_dataframe_pickle(df)
