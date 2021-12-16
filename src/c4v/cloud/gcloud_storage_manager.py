"""
    Cloud storage manager object, to easily retrieve and upload files from and to google cloud storage
"""
# Internal imports
from c4v.cloud.utils import make_tarfile

# External imports
from google.cloud import storage
import enum

# Python imports
from io import BytesIO
from typing import List
import tempfile
import tarfile
import pathlib
import logging

class ClassifierType(enum.Enum):
    """
        Possible types of classifiers
    """
    RELEVANCE : str = "relevance"
    SERVICE   : str = "service"

class GCSStorageManager:
    """
    Manage data stored un Google Cloud Storage. It gives you access
    to the classifier to use during classifications
    
    # Parameters:
        - bucket : `str` = Name of the bucket to interact with.
        - bucket_prefix_path : `str` = Path inside the provided bucket where files will be looked for
    """

    def __init__(self, bucket_name: str, bucket_prefix_path: str):
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)
        self._bucket_prefix_path = bucket_prefix_path

    @property
    def bucket(self) -> storage.Bucket:
        """
            Name of the bucket to interact with
        """
        return self._bucket

    @property
    def bucket_prefix_path(self) -> str:
        """
            Path inside the provided bucket where files will be looked for
        """
        return self._bucket_prefix_path

    def download_classifier_model_to(self, type : ClassifierType, filepath : str):
        """
            Try to download the classifier model of type 'type' to the provided filepath in local storage

            # Parameters:
                - type : `ClassifierType` = type of classifier to get
                - filepath : `str` = where to store it in local storage
        """

        # Create name of file, use the prefix path (the path where to look for files) 
        # and find the file as a tar file
        blob_name = f"{self.bucket_prefix_path}/{type.value}.tar"

        # Get blob and download it to local storage
        blob = self.bucket.get_blob(blob_name)

        # Create a templfile where to download the tar file
        with tempfile.NamedTemporaryFile() as file:
            logging.info(f"Retrieving file: {blob_name} from bucket {self.bucket.name}")
            blob.download_to_file(file)

            # untar that file to the desired path
            logging.info(f"Extracting tar file content to '{filepath}'...")
            tar = tarfile.TarFile(file.name)
            tar.extractall(filepath)

    def upload_classifier_model_from(self, type : ClassifierType, filepath : str):
        """
            Try to upload the classifier model of type 'type' stored in the provided filepath in local storage

            # Parameters:
                - type : `ClassifierType` = type of classifier to get
                - filepath : `str` = where to find it in local storage
        """
        # Check if the file exists beforehand 
        if not pathlib.Path(filepath).exists():
            raise ValueError(f"Path in filepath '{filepath}' does not exists")

        # Create name of file, use the prefix path (the path where to look for files) 
        # and find the file as a tar file
        blob_name = f"{self.bucket_prefix_path}/{type.value}.tar"
        logging.info(f"Creating tar file for file: {filepath}...")

        # Create a temp file where to tar the folder
        with tempfile.NamedTemporaryFile() as file:
            make_tarfile(file.name, filepath)
            logging.info(f"Uploading to blob: '{blob_name}' from local storage '{filepath}' to bucket: '{self.bucket.name}'")
            # Get blob and upload to it from local storage
            blob = self.bucket.get_blob(blob_name)
            blob.upload_from_filename(file.name)

    def get_byte_stream(self, filepath: str) -> BytesIO:
        """
            Download a file to bytestream object
            # Parameters:
                - filepath : `str` = path to file inside the bucket to download
            # Return:
                A BytesIO object holding the entirely downloaded object 
        """
        blob_name = filepath if filepath.startswith(self.bucket_prefix_path) \
            else f"{self.bucket_prefix_path}/{filepath}"
        logging.info(f"Bucket {self.bucket.name}: Retrieving file: {blob_name}")
        blob = self.bucket.get_blob(blob_name)
        byte_stream = BytesIO()
        blob.download_to_file(byte_stream)
        byte_stream.seek(0)
        return byte_stream

    def save_file(self, destination_file : str, source_file : str):
        """
            Save a file into the given path 
            # Parameters:
                - destination_file : `str` = path inside the bucket where to store the uploaded file
                - source_file : `str` = path in local storage to the file that will be uploaded
        """
        blob_name = f"{self.bucket_prefix_path}/{destination_file}"
        logging.info(f"Bucket {self.bucket.name}: Saving file with name: {blob_name}")
        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(source_file)

    def list_files(self, prefix_path: str = None, delimiter: str = None) -> List[str]:
        """
            List files inside the bucket under the given path
            # Parameters:
                - prefix_path : `str` = (optional) path where the files will be looked for; relative to this object's prefix path
            # Return:
                List of filenames of stored objects
        """
        full_prefix = self.bucket_prefix_path if prefix_path is None else f"{self.bucket_prefix_path}/{prefix_path}"
        logging.info(f"Bucket {self.bucket.name}: Listing files with prefix: {full_prefix}")
        blobs = self._client.list_blobs(self.bucket, prefix=full_prefix, delimiter=delimiter)
        return [blob.name for blob in blobs]
