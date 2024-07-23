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

# define models that could be used to generate embeddings
# e.g. prott5, esm2
hugging_dict = {
    "prott5": {"model_name": r'Rostlab/prot_t5_xl_uniref50',
               "model_type": T5EncoderModel,
               "tokenizer": T5Tokenizer}
}


class ProtSeqEmbedder:
    """
    Class to handle embedding generation, storage and retrieval based on input path.
    """

    def __init__(self, model: Literal['prott5', 'esm']):
        self.model_conf = hugging_dict.get(model)
        self.model_name = self.model_conf.get('model_name')
        self.output_path = os.path.join(os.getcwd(), "input_data/raw_embeddings")

    def _init_huggingface(self):

        self.model = self.model_conf["model_type"].from_pretrained(self.model_name)
        self.tokenizer = self.model_conf["tokenizer"].from_pretrained(self.model_name, do_lower_case=False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_raw_embeddings(self, path_to_fasta: str, pp: bool = True) -> list[tuple[str, torch.tensor]]:
        """
        This method returns the raw embeddings of protein sequences. If the fasta file has been converted to embeddings before, the
        embeddings are read from the h5 file.
        Identification is done by checking the hash over: first 10 Seqs, first 10 Ids, model name and file length
        Args:
            path_to_fasta: Path the fasta file that shall be read
            pp: Whether to load per protein embeddings

        Returns:
            A list containing the id and the embedding for each sequence
        """
        embed_type = 'pp' if pp else 'pa'
        logger.debug(f'Getting {embed_type} embed')
        fasta_seqs = []
        fasta_ids = []

        with open(path_to_fasta, 'r') as fasta:
            # get file length for bette file uniqueness criteria
            file_length = self._get_filelength(fasta)
            # read the fasta sequence
            for i, record in enumerate(SeqIO.parse(fasta, 'fasta')):
                fasta_seqs.append(str(record.seq))
                fasta_ids.append(record.id)

                # check if this fasta file has been seen before and if there are existing embeddings
                if i == 10:
                    # crate a hash across first 10 seqs, 10 ids, the model name and the file length
                    bof_hash = sha1(''.join(fasta_seqs + fasta_ids + [self.model_name,str(file_length)]).encode()).hexdigest()[:10]
                    # if there is a dir with the hash as its name read from that dir
                    if bof_hash in self._get_h5_dirnames():
                        embedfile = os.path.join(self.output_path, bof_hash, embed_type + '.h5')
                        logger.debug(f'Found precomputed embedding file at {embedfile}')
                        return self._read_h5_embeddings(embedfile)
                    logger.debug('No precomputed embedding file found')

        # In case no existing embeddings were found
        # init all the necessary huggingface model
        self._init_huggingface()

        # create new dir and wire pp and pa files
        embedd_output_dir = os.path.join(self.output_path, bof_hash)
        os.mkdir(embedd_output_dir)
        for seq_id, seq in zip(fasta_ids, fasta_seqs):
            embedding = self._create_embeddings([seq])
            # store the per protein embedding
            self.store_in_h5(header=seq_id, embedding_tensors=embedding.cpu().mean(axis=0), filename=os.path.join(embedd_output_dir, 'pp.h5'))
            # store the per residue embedding
            self.store_in_h5(header=seq_id, embedding_tensors=embedding.cpu(), filename=os.path.join(embedd_output_dir, 'pa.h5'))

        logger.debug('Wrote embedding files adding meta data')
        # add a json file with metadata to dir so its human-readable
        with open(os.path.join(embedd_output_dir, 'meta.json'), 'w') as meta:
            json.dump({
                "fasta_file": path_to_fasta,
                "model": self.model_name,
                "date": str(datetime.now()),
                "num_seqs": len(fasta_seqs),
                "size": file_length
            }, meta, indent=4)

        embedfile = os.path.join(self.output_path, bof_hash, embed_type + '.h5')
        # return reading the previously written files
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
        Read the h5 prott5_embeddings from a h5 file and return them as a list of tuples.
        Args:
            h5_filepath: The path to the h5 file

        Returns:
            A list of tuples containing the id and the embedding for each sequence
        """
        embeddings = []
        embedd_file = h5py.File(h5_filepath)
        pbar_desc = f'Loading prott5_embeddings'
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

    def _get_filelength(self, file_handle):
        current_position = file_handle.tell()  # Remember the current position in the file
        file_handle.seek(0, 2)  # Go to the end of the file
        file_length = file_handle.tell()  # Get the position at the end (which is the file length)
        file_handle.seek(current_position)  # Go back to the original position
        return file_length


if __name__ == '__main__':
    protseq_embedder = ProtSeqEmbedder('prott5')
    test_embedd = protseq_embedder.get_raw_embeddings(
        '/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/input_data/input_case/test1/phosphatase.fasta')

    print('hi')
