import concurrent
import hashlib
import os.path
import urllib

import requests
import logging
import logging.config
import time
from concurrent.futures import ThreadPoolExecutor,wait

from Bio import SeqIO

logging.config.fileConfig(
    '/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/logging.config',
    disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class DataPreper:
    """
    This class queries interpro for the best fitting SUPERFAMILY label
    """

    def __init__(self):
        # the url to the seqscan endpoint
        self.interpro_url = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"

    def summit_search(self, sequence: str) -> str:
        """
        Queries the interpro sequence search endpoint with the input sequence
        Args:
            sequence: The query sequence

        Returns:
            The inter pro result
        """
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        # use the default email to get back a job id
        data = {"sequence": sequence,
                "email": "interpro-team@ebi.ac.uk"}

        # Submit the job
        resp = requests.post(f'{self.interpro_url}/run', headers=headers, data=data)

        return resp.text

    def check_status(self, jobname: str) -> bool:
        """
        Check the status of a running job
        Args:
            jobname: The name of the job

        Returns:
            If the job has finished
        """
        resp = requests.get(f"{self.interpro_url}/status/{jobname}")
        status = resp.text
        return True if status == "FINISHED" else False

    def get_resulting_family(self, jobname: str) -> str:
        """
        Given the name of a completed job, query the results for the FAMILY label
        Args:
            jobname: The name of the job

        Returns:
            The best found FAMILY label
        """
        resp = requests.get(f"{self.interpro_url}/result/{jobname}/json")
        resp_data = resp.json()
        # iterate over the json job result
        for match in resp_data["results"][0]["matches"]:
            try:
                # if there is a FAMILY entry use it
                if match["signature"]["entry"]["accession"] is not None and match["signature"]["entry"]["type"] == "FAMILY":
                    return match["signature"]["entry"]["accession"]
            except (TypeError,AttributeError) as e:
                continue
        # incase no FAMILY was found
        return "NoFamFound"

    def enhance_fasta(self, path_to_fasta: str):
        """
        Given an input fasta file, this method tries to augment the fasta labels with the best fitting Inter PRO FAMILY label
        Args:
            path_to_fasta: The path to the input fasta file

        Returns:
            None, wires a new enhanced fasta file in the same directory as the input fasta file
        """
        assert os.path.isfile(path_to_fasta), "File does not exist"

        # submit all search jobs to interpro
        _cc_jobs = []
        seqrec_jobs = []
        completed_jobs = 0
        # multithread the retrieval
        with ThreadPoolExecutor(max_workers=16) as executor:
            # iterate over the fasta records
            for seqrec in SeqIO.parse(path_to_fasta, "fasta"):
                cc_job = executor.submit(self.summit_search,str(seqrec.seq))
                seqrec_jobs.append({"seqrec": seqrec,
                                    "job_id": cc_job,
                                    "family": "",
                                    "finished": False})
                _cc_jobs.append(cc_job)
                time.sleep(0.5)
        # wait for all interpro jobs to be submitted
        wait(_cc_jobs)

        time.sleep(120)
        # iterate over all jobs and see if they are finished
        while completed_jobs < len(seqrec_jobs):

            for seqrec_job in seqrec_jobs:
                # Dont check finished jobs
                if seqrec_job["finished"]:
                    continue
                # check if the job is done
                if self.check_status(seqrec_job["job_id"].result()):
                    seqrec_job["family"] = self.get_resulting_family(seqrec_job["job_id"].result())

                    seqrec_job["finished"] = True
                    completed_jobs += 1

        # wire the data

        with open(path_to_fasta.rstrip('*.fasta') + '_enhanced.fasta', "w") as fasta_file:
            for seqrec_job in seqrec_jobs:
                seqrec = seqrec_job["seqrec"]
                seqrec.id = seqrec.id + '_' + seqrec_job["family"]
                seqrec.description = ""

                SeqIO.write(seqrec,fasta_file, "fasta")


if __name__ == '__main__':
    test_prepper = DataPreper()
    test_prepper.enhance_fasta("/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/input_data/input_case/kinase/kinase_porcessiing.fa")