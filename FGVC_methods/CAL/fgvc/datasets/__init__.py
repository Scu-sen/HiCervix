from .aircraft_dataset import AircraftDataset
from .bird_dataset import BirdDataset
from .car_dataset import CarDataset
from .tct_dataset import get_dataset

# def get_trainval_datasets(tag, resize):
#     if tag == 'aircraft':
#         return AircraftDataset(phase='train', resize=resize), AircraftDataset(phase='val', resize=resize)
#     elif tag == 'bird':
#         return BirdDataset(phase='train', resize=resize), BirdDataset(phase='val', resize=resize)
#     elif tag == 'car':
#         return CarDataset(phase='train', resize=resize), CarDataset(phase='val', resize=resize)
#     elif tag == 'tct':
#         #return TCTDataset(phase='train', resize=resize), TCTDataset(phase='val', resize=resize)
#         return trainset, testset
#     else:
#         raise ValueError('Unsupported Tag {}'.format(tag))

def get_trainval_datasets(config):
    if config.tag == 'tct':
        #return TCTDataset(phase='train', resize=resize), TCTDataset(phase='val', resize=resize)
        return get_dataset(config)
    else:
        raise ValueError('Unsupported Tag {}'.format(tag))