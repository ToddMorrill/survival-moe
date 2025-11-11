import torch
from survkit.data import generate_label_sequence

from .test_utils import get_config

def test_MNISTSurvival_generate_label_sequence():
    _, train_config = get_config("--time_bins 3")
    # test a time before the first bin, between bins, at a bin mark, and after the last bin 
    time_bins = torch.tensor([0.0, 0.2500, 2.2237, 4.1974])
    event_times = torch.tensor([-1, 0.05, 1.0577, 2.2237, 5.3900])
    floor_labels_expected = torch.tensor([[1, 1, 1, 1],
                                          [1, 1, 1, 1],
                                          [0, 1, 1, 1],
                                          [0, 0, 1, 1],
                                          [0, 0, 0, 1]])
    floor_label_sequences = generate_label_sequence(event_times, time_bins, floor=True)
    assert torch.equal(floor_label_sequences, floor_labels_expected)
    ceil_labels_expected = torch.tensor([[1, 1, 1, 1],
                                         [0, 1, 1, 1],
                                         [0, 0, 1, 1],
                                         [0, 0, 1, 1],
                                         [0, 0, 0, 0]])
    ceil_label_sequences = generate_label_sequence(event_times, time_bins, floor=False)
    assert torch.equal(ceil_label_sequences, ceil_labels_expected)

    
