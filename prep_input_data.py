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

    def __init__(self):
        self.interpro_url = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"

    def summit_search(self, sequence: str) -> str:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {"sequence": sequence,
                "email": "interpro-team@ebi.ac.uk"}

        # Submit the job
        resp = requests.post(f'{self.interpro_url}/run', headers=headers, data=data)

        return resp.text

    def check_status(self, jobname: str) -> bool:
        resp = requests.get(f"{self.interpro_url}/status/{jobname}")
        status = resp.text
        return True if status == "FINISHED" else False

    def get_resulting_family(self, jobname: str) -> str:
        resp = requests.get(f"{self.interpro_url}/result/{jobname}/json")
        resp_data = resp.json()
        for match in resp_data["results"][0]["matches"]:
            try:
                if match["signature"]["entry"]["accession"] is not None and match["signature"]["entry"]["type"] == "FAMILY":
                    return match["signature"]["entry"]["accession"]
            except (TypeError,AttributeError) as e:
                continue

        return "NoFamFound"

    def enhance_fasta(self, path_to_fasta: str):
        assert os.path.isfile(path_to_fasta), "File does not exist"

        # submit all search jobs to interpro
        _cc_jobs = []
        seqrec_jobs = []
        completed_jobs = 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            for seqrec in SeqIO.parse(path_to_fasta, "fasta"):
                cc_job = executor.submit(self.summit_search,str(seqrec.seq))
                seqrec_jobs.append({"seqrec": seqrec,
                                    "job_id": cc_job,
                                    "family": "",
                                    "finished": False})
                _cc_jobs.append(cc_job)
                time.sleep(0.5)
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

        # wirte the data

        with open(path_to_fasta.rstrip('*.fasta') + '_enhanced.fasta', "w") as fasta_file:
            for seqrec_job in seqrec_jobs:
                seqrec = seqrec_job["seqrec"]
                seqrec.id = seqrec.id + '_' + seqrec_job["family"]
                seqrec.description = ""

                SeqIO.write(seqrec,fasta_file, "fasta")

    def test(self, sequence: str) -> str:

        job = self.summit_search(sequence)
        while not self.check_status(job):
            time.sleep(1)
        print(self.get_resulting_family(job))


if __name__ == '__main__':
    test_prepper = DataPreper()
    test_prepper.enhance_fasta("/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/input_data/input_case/KLK/mini.fasta")