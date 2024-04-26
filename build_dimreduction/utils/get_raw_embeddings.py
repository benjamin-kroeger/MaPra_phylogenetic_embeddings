import json
import logging.config
import os
import re
from datetime import datetime
from glob import glob
from hashlib import sha1
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

logging.config.fileConfig(
    '/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/logging.config',
    disable_existing_loggers=False)
logger = logging.getLogger(__name__)

hugging_dict = {
    "prott5": {"model_name": r'Rostlab/prot_t5_xl_uniref50',
               "model_type": T5EncoderModel,
               "tokenizer": T5Tokenizer}
}


class ProtSeqEmbedder:

    def __init__(self, model: Literal['prott5', 'esm']):
        model_conf = hugging_dict.get(model)
        self.model_name = model_conf.get('model_name')
        self.model = model_conf["model_type"].from_pretrained(self.model_name)
        self.tokenizer = model_conf["tokenizer"].from_pretrained(self.model_name, do_lower_case=False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.output_path = os.path.join(os.getcwd(),"input_data/raw_embeddings")

    def get_raw_embeddings(self, path_to_fasta: str, skip_fasta_loading: bool = False, pp: bool = True) -> list[tuple[str, torch.tensor]]:
        """
        This method returns the raw embeddings of protein sequences. If the fasta file has been converted to embeddings before, the
        embeddings are read from the h5 file.
        Identification is done by checking the hash over: first 10 Seqs, first 10 Ids, Modelname
        Args:
            path_to_fasta: Path the fasta file that shall be read
            skip_fasta_loading:
            pp: Whether to load per protein embeddings

        Returns:
            A list containing the id and the embedding for each sequence
        """
        embed_type = 'pp' if pp else 'pa'
        logger.debug(f'Getting {embed_type} embed')
        fasta_seqs = []
        fasta_ids = []

        with open(path_to_fasta, 'r') as fasta:
            for i, record in enumerate(SeqIO.parse(fasta, 'fasta')):
                fasta_seqs.append(str(record.seq))
                fasta_ids.append(record.id)

                # check if this fasta file has been seen before and if there are existing embeddings
                if i == 10:
                    bof_hash = sha1(''.join(fasta_seqs + fasta_ids + [self.model_name]).encode()).hexdigest()[:10]
                    if bof_hash in self._get_h5_dirnames():
                        embedfile = os.path.join(self.output_path, bof_hash, embed_type + '.h5')
                        logger.debug(f'Found precomputed embedding file at {embedfile}')
                        return self._read_h5_embeddings(embedfile)
                    logger.debug('No precomputed embedding file found')

        # create new dir and wirte pp and pa files
        embedd_output_dir = os.path.join(self.output_path, bof_hash)
        os.mkdir(embedd_output_dir)
        for seq_id, seq in zip(fasta_ids, fasta_seqs):
            embedding = self._create_embeddings([seq])
            self.store_in_h5(header=seq_id, embedding_tensors=embedding.cpu().mean(axis=0), filename=os.path.join(embedd_output_dir, 'pp.h5'))
            self.store_in_h5(header=seq_id, embedding_tensors=embedding.cpu(), filename=os.path.join(embedd_output_dir, 'pa.h5'))

        logger.debug('Wrote embedding files adding meta data')
        with open(os.path.join(embedd_output_dir, 'meta.json'), 'w') as meta:
            json.dump({
                "fasta_file": path_to_fasta,
                "model": self.model_name,
                "date": str(datetime.now()),
                "size": len(fasta_seqs)
            }, meta, indent=4)

        embedfile = os.path.join(self.output_path, bof_hash, embed_type + '.h5')
        return self._read_h5_embeddings(embedfile)

    def _get_h5_dirnames(self) -> list[str]:
        """
        Get all folders containing h5 files.
        Returns:
            A list of folder names
        """
        return [os.path.basename(x) for x in glob(os.path.join(self.output_path, '*'))]

    def _read_h5_embeddings(self, h5_filepath) -> list[tuple[str, torch.tensor]]:
        """
        Read the h5 embeddings from a h5 file and return them as a list of tuples.
        Args:
            h5_filepath: The path to the h5 file

        Returns:
            A list of tuples containing the id and the embedding for each sequence
        """
        embeddings = []
        embedd_file = h5py.File(h5_filepath)
        pbar_desc = f'Loading embeddings'
        with tqdm(total=len(embedd_file.keys()), desc=pbar_desc) as pbar:
            for key in embedd_file.keys():
                embedding = np.array(embedd_file[key])
                embedding = torch.tensor(embedding).to(torch.float32)
                embeddings.append((key, embedding))
                pbar.update(1)

        return embeddings

    def _create_embeddings(self, sequences: list) -> torch.tensor:

        # clean seqs
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
        ids = self.tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
        model = self.model.to(self.device)
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        return_tensor = []
        for i in range(len(sequences)):
            return_tensor.append(embedding_repr.last_hidden_state[i, :sum(attention_mask[i])])

        return torch.stack(return_tensor)[0]

    def store_in_h5(self, header, embedding_tensors, filename):

        if Path(filename).is_file():
            # print("Appending to file...")
            h5f = h5py.File(filename, 'a')
        else:
            # print("Making file...")
            h5f = h5py.File(filename, 'w')

        h5f.create_dataset(header, data=embedding_tensors)
        h5f.close()


if __name__ == '__main__':
    protseq_embedder = ProtSeqEmbedder('prott5')
    test_embedd = protseq_embedder.get_raw_embeddings('/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/foo.fasta')

    print('hi')
