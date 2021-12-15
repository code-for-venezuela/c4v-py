from google.cloud import storage
import enum
from io import BytesIO

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

    bucket : `str`
        The bucket name to interact with.
    bucket_prefix : `str`
        The prefix for all interactions with the bucket.
    """

    def __init__(self, bucket: str, bucket_prefix: str):
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)
        self._bucket_prefix = bucket_prefix + "/scraped_data"

    @property
    def bucket(self) -> storage.Bucket:
        return self._bucket

    @property
    def bucket_prefix(self) -> str:
        return self._bucket_prefix

    def download_classifier_model_to(self, type : ClassifierType, filepath : str):
        """
            Try to download the classifier model of type 'type' to the provided filepath in local storage

            # Parameters:
                - type : `ClassifierType` = type of classifier to get
                - filepath : `str` = where to store it in local storage
        """
        raise NotImplementedError("Not yet implemented")

    def upload_classifier_model_from(self, type : ClassifierType, filepath : str):
        """
            Try to upload the classifier model of type 'type' stored in the provided filepath in local storage

            # Parameters:
                - type : `ClassifierType` = type of classifier to get
                - filepath : `str` = where to find it in local storage
        """
        raise NotImplementedError("Not yet implemented")


    def get_byte_stream(self, filepath: str):
        blob_name = filepath if filepath.startswith(self.bucket_prefix) \
            else f"{self.bucket_prefix}/{filepath}"
        print(f"Bucket {self.bucket.name}: Retrieving file: {blob_name}")
        blob = self.bucket.get_blob(blob_name)
        byte_stream = BytesIO()
        blob.download_to_file(byte_stream)
        byte_stream.seek(0)
        return byte_stream


    def save_file(self, destination_file, csv_source_file):
        blob_name = f"{self.bucket_prefix}/{destination_file}"
        print(f"Bucket {self.bucket.name}: Saving file with name: {blob_name}")
        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(csv_source_file)


    def list_files(self, prefix: str = None, delimiter: str = None):
        full_prefix = self.bucket_prefix if prefix is None else f"{self.bucket_prefix}/{prefix}"
        print(f"Bucket {self.bucket.name}: Listing files with prefix: {full_prefix}")
        blobs = self._client.list_blobs(self.bucket, prefix=full_prefix, delimiter=delimiter)
        return [blob.name for blob in blobs]
