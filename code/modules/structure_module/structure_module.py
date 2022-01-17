import numpy as np

from modules.structure_module.predict import get_sections
from utilities.feature_extractor import FeatureExtractor


def get_structure(file):
    ex = FeatureExtractor()
    try:
        sections = get_sections(file)
        structure = ex.get_intensity(file, sections)
    except:
        structure = (np.array([[0, 0, 1, 0]]), 1)
    if structure[0].shape[0] <= 1:
        structure = (np.array([[0, 0, 1, 0]]), 1)
    return structure