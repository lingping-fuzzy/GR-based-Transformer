"""
    File to load dataset based on user control from main file
"""
from data.molecules import MoleculeDataset
from data.SBMs import SBMsDataset
from data.TUs import TUsDataset
from data.cycles import CyclesDataset
from data.COLLAB import COLLABDataset
from data.WikiCS import WikiCSDataset

def LoadData(DATASET_NAME):
    """
        This function is called in the main_xx.py file 
        returns:
        ; dataset object
    """    
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ZINC':
        return MoleculeDataset(DATASET_NAME)

    # handling for SBM datasets
    SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    if DATASET_NAME in SBM_DATASETS:

        return SBMsDataset(DATASET_NAME)

    # handling for the TU Datasets
    TU_DATASETS = ['ENZYMES', 'DD', 'PROTEINS_full', 'MUTAG', 'PROTEINS', 'KKI', 'OHSU', 'Peking_1']
    if DATASET_NAME in TU_DATASETS:
        return TUsDataset(DATASET_NAME)

    CL_DATASETS = ['CYCLES']
    if DATASET_NAME in CL_DATASETS:
        return CyclesDataset(DATASET_NAME)

    # handling for COLLAB dataset
    if DATASET_NAME == 'OGBL-COLLAB':
        return COLLABDataset(DATASET_NAME)

    if DATASET_NAME == 'WikiCS':
        return WikiCSDataset(DATASET_NAME)
    