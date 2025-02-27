"""
Google Cloud Storage integration for the Personal Research Assistant.
"""
import logging
import os
from typing import Optional, BinaryIO

from google.cloud import storage

from config import get_config

# Get configuration
config = get_config()

# Set up logging
logger = logging.getLogger(__name__)

def get_storage_client():
    """
    Get the Google Cloud Storage client.
    
    Returns:
        storage.Client: Storage client
    """
    try:
        # Check for credentials
        if not config.google.credentials_path:
            logger.error("Google Cloud credentials path not found")
            raise ValueError("Google Cloud credentials path not found in configuration")
            
        # Set environment variable for credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.google.credentials_path
        
        # Create client
        client = storage.Client()
        
        return client
        
    except Exception as e:
        logger.error(f"Error creating Storage client: {str(e)}")
        raise

def get_bucket(bucket_name: Optional[str] = None):
    """
    Get a Google Cloud Storage bucket.
    
    Args:
        bucket_name: Name of the bucket (uses config if not provided)
        
    Returns:
        storage.Bucket: Storage bucket
    """
    # Use bucket name from config if not provided
    bucket_name = bucket_name or config.google.storage_bucket
    
    if not bucket_name:
        logger.error("Google Cloud Storage bucket name not found")
        raise ValueError("Google Cloud Storage bucket name not found in configuration")
        
    try:
        # Get client
        client = get_storage_client()
        
        # Get bucket
        bucket = client.bucket(bucket_name)
        
        # Create bucket if it doesn't exist
        if not bucket.exists():
            logger.info(f"Creating bucket: {bucket_name}")
            bucket.create()
        
        return bucket
        
    except Exception as e:
        logger.error(f"Error getting bucket: {str(e)}")
        raise

def upload_file(file_path: str, destination_blob_name: Optional[str] = None, bucket_name: Optional[str] = None):
    """
    Upload a file to Google Cloud Storage.
    
    Args:
        file_path: Path to the file to upload
        destination_blob_name: Name to give the file in the bucket (uses filename if not provided)
        bucket_name: Name of the bucket (uses config if not provided)
        
    Returns:
        str: Public URL of the uploaded file
    """
    logger.info(f"Uploading file: {file_path}")
    
    # Use filename as destination if not provided
    if not destination_blob_name:
        destination_blob_name = os.path.basename(file_path)
    
    try:
        # Get bucket
        bucket = get_bucket(bucket_name)
        
        # Create blob
        blob = bucket.blob(destination_blob_name)
        
        # Upload file
        blob.upload_from_filename(file_path)
        
        logger.info(f"File {file_path} uploaded to {destination_blob_name}")
        
        # Make public
        blob.make_public()
        
        # Return public URL
        return blob.public_url
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise

def upload_from_memory(data: bytes, destination_blob_name: str, content_type: Optional[str] = None, bucket_name: Optional[str] = None):
    """
    Upload data from memory to Google Cloud Storage.
    
    Args:
        data: Data to upload
        destination_blob_name: Name to give the file in the bucket
        content_type: Content type of the data
        bucket_name: Name of the bucket (uses config if not provided)
        
    Returns:
        str: Public URL of the uploaded file
    """
    logger.info(f"Uploading data to: {destination_blob_name}")
    
    try:
        # Get bucket
        bucket = get_bucket(bucket_name)
        
        # Create blob
        blob = bucket.blob(destination_blob_name)
        
        # Upload data
        blob.upload_from_string(data, content_type=content_type)
        
        logger.info(f"Data uploaded to {destination_blob_name}")
        
        # Make public
        blob.make_public()
        
        # Return public URL
        return blob.public_url
        
    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}")
        raise

def download_file(source_blob_name: str, destination_file_path: str, bucket_name: Optional[str] = None):
    """
    Download a file from Google Cloud Storage.
    
    Args:
        source_blob_name: Name of the file in the bucket
        destination_file_path: Path to save the file to
        bucket_name: Name of the bucket (uses config if not provided)
    """
    logger.info(f"Downloading file: {source_blob_name}")
    
    try:
        # Get bucket
        bucket = get_bucket(bucket_name)
        
        # Create blob
        blob = bucket.blob(source_blob_name)
        
        # Download file
        blob.download_to_filename(destination_file_path)
        
        logger.info(f"File {source_blob_name} downloaded to {destination_file_path}")
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise

def delete_file(blob_name: str, bucket_name: Optional[str] = None):
    """
    Delete a file from Google Cloud Storage.
    
    Args:
        blob_name: Name of the file in the bucket
        bucket_name: Name of the bucket (uses config if not provided)
    """
    logger.info(f"Deleting file: {blob_name}")
    
    try:
        # Get bucket
        bucket = get_bucket(bucket_name)
        
        # Create blob
        blob = bucket.blob(blob_name)
        
        # Delete file
        blob.delete()
        
        logger.info(f"File {blob_name} deleted")
        
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise

def list_files(prefix: Optional[str] = None, bucket_name: Optional[str] = None):
    """
    List files in Google Cloud Storage.
    
    Args:
        prefix: Prefix to filter files by
        bucket_name: Name of the bucket (uses config if not provided)
        
    Returns:
        list: List of file names
    """
    logger.info(f"Listing files with prefix: {prefix}")
    
    try:
        # Get bucket
        bucket = get_bucket(bucket_name)
        
        # List blobs
        blobs = bucket.list_blobs(prefix=prefix)
        
        # Get blob names
        blob_names = [blob.name for blob in blobs]
        
        logger.info(f"Found {len(blob_names)} files")
        
        return blob_names
        
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise
