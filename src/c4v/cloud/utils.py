"""
    Utility functions for the cloud module
"""
import tarfile
import os.path

def make_tarfile(output_filename : str, source_dir : str):
    """
        Create a tar file from the given directory to the specified output file
        # Parameters
            - output_filename : `str` = Name of the file to be written
            - source_dir : `str` = path to the directory in local storage to be tared
    """
    with tarfile.open(output_filename, "w") as tar:
        # this arcname arguments prevents from this tar to include the local filesystem layout
        tar.add(source_dir, arcname=os.path.basename(source_dir))

