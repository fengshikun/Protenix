import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset, NestedDictionaryDataset, FromNumpyDataset, Dictionary, PrependTokenDataset, AppendTokenDataset, RightPadDataset2D, RightPadDataset, RawArrayDataset
import torch


class PrependAndAppend2DDataset(BaseWrapperDataset):
    def __init__(self, dataset, token=None):
        super().__init__(dataset)
        self.token = token

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.token is not None:
            h, w = item.size(-2), item.size(-1)
            new_item = torch.full((h + 2, w + 2), self.token).type_as(item)
            new_item[1:-1, 1:-1] = item
            return new_item
        return item

class DistanceDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        pos = self.dataset[idx].view(-1, 3)        
        dist = torch.cdist(pos, pos, p=2)
        # dist = distance_matrix(pos, pos).astype(np.float32)
        return dist

class EdgeTypeDataset(BaseWrapperDataset):
    def __init__(self, dataset: torch.utils.data.Dataset, num_types: int):
        self.dataset = dataset
        self.num_types = num_types

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        node_input = self.dataset[index].clone()
        offset = node_input.view(-1, 1) * self.num_types + node_input.view(1, -1)
        return offset


class TokenizeDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dictionary: Dictionary,
        max_seq_len: int=512,
    ):
        self.dataset = dataset
        self.dictionary = dictionary
        self.max_seq_len = max_seq_len

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        raw_data = self.dataset[index]
        # Be tolerant to malformed/edge-case ligands to keep long training runs alive.
        if len(raw_data) == 0:
            return torch.tensor([self.dictionary.unk()], dtype=torch.long)
        elif len(raw_data) >= self.max_seq_len:
            raw_data = raw_data[: self.max_seq_len - 1]
        return torch.from_numpy(self.dictionary.vec_index(raw_data)).long()

class KeyDataset(BaseWrapperDataset):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.dataset[idx][self.key]

class LengthDataset(BaseWrapperDataset):

    def __init__(self, dataset):
        super().__init__(dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return len(item)

class NormalizeDataset(BaseWrapperDataset):
    def __init__(self, dataset, coordinates, normalize_coord=True):
        self.dataset = dataset
        self.coordinates = coordinates
        self.normalize_coord = normalize_coord  # normalize the coordinates.
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        coordinates = dd[self.coordinates]
        # normalize
        if self.normalize_coord:
            coordinates = coordinates - coordinates.mean(axis=0)
            dd[self.coordinates] = coordinates
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class RemoveHydrogenDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        atoms,
        coordinates,
        remove_hydrogen=False,
        remove_polar_hydrogen=False,
    ):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.remove_hydrogen = remove_hydrogen
        self.remove_polar_hydrogen = remove_polar_hydrogen
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        atoms = dd[self.atoms]
        coordinates = dd[self.coordinates]
        raw_atoms = atoms
        raw_coordinates = coordinates

        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            atoms = atoms[mask_hydrogen]
            #print(coordinates.shape)
            coordinates = coordinates[mask_hydrogen]
            # Some ligands can become empty after hydrogen filtering.
            # Fall back to raw arrays to avoid downstream empty-token crashes.
            if len(atoms) == 0:
                atoms = raw_atoms
                coordinates = raw_coordinates
        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms = atoms[:-end_idx]
                coordinates = coordinates[:-end_idx]
                if len(atoms) == 0:
                    atoms = raw_atoms
                    coordinates = raw_coordinates
        dd[self.atoms] = atoms
        dd[self.coordinates] = coordinates
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)







def load_mols_dataset(dataset_dict, dictionary, max_seq_len=512):
    def PrependAndAppend(dataset, pre_token, app_token):
        dataset = PrependTokenDataset(dataset, pre_token)
        return AppendTokenDataset(dataset, app_token)



    dataset = RemoveHydrogenDataset(dataset_dict, "atoms", "coordinates", True, True)


    apo_dataset = NormalizeDataset(dataset, "coordinates")

    src_dataset = KeyDataset(apo_dataset, "atoms")
    len_dataset = LengthDataset(src_dataset)
    src_dataset = TokenizeDataset(
        src_dataset, dictionary, max_seq_len=max_seq_len
    )
    coord_dataset = KeyDataset(apo_dataset, "coordinates")
    src_dataset = PrependAndAppend(
        src_dataset, dictionary.bos(), dictionary.eos()
    )
    edge_type = EdgeTypeDataset(src_dataset, len(dictionary))
    # coord_dataset = FromNumpyDataset(coord_dataset)
    distance_dataset = DistanceDataset(coord_dataset)
    coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
    distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


    nest_dataset = NestedDictionaryDataset(
        {
            "net_input": {
                "mol_src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=dictionary.pad(),
                ),
                "mol_src_distance": RightPadDataset2D(
                    distance_dataset,
                    pad_idx=0,
                ),
                "mol_src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
            },
            # "smi_name": RawArrayDataset(smi_dataset),
            # "target":  RawArrayDataset(label_dataset),
            "mol_len": RawArrayDataset(len_dataset),
        },
    )
    return nest_dataset
