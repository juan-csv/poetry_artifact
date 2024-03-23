# pip install google-cloud-storage
from io import StringIO
import pandas as pd
from google.cloud import storage

"""
path_key = "interno-davinci-analitica-bf3c6a5a0111.json"
# instanciar cliente
storage_client = storage.Client.from_service_account_json(path_key)
"""


def upload_df_to_gcs(bucket_name: str, destination_blob_name: str, df: pd.DataFrame) -> bool:
    """
    Uploads a DataFrame to a Google Cloud Storage (GCS) bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        destination_blob_name (str): The name of the destination blob.
        df (pandas.DataFrame): The DataFrame to be uploaded.

    Returns:
        bool: True if the upload was successful, False otherwise.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Convierte el DataFrame a CSV en memoria usando StringIO
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    # Mueve el puntero al inicio del objeto StringIO para leer su contenido
    csv_buffer.seek(0)

    # Sube el contenido del objeto StringIO al bucket de GCS
    blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')

    return True


def upload_blob(bucket_name: str, source_file_name: str, destination_blob_name: str) -> bool:
    """
    Uploads a file to a specified cloud storage bucket.

    Args:
        bucket_name (str): The name of the cloud storage bucket.
        source_file_name (str): The name of the file to be uploaded.
        destination_blob_name (str): The name of the destination blob.

    Returns:
        bool: True if the file was successfully uploaded, False otherwise.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"File: {source_file_name} upload to {destination_blob_name}.")

    return True


def download_blob(bucket_name: str, source_blob_name: str, destination_file_name: str) -> bool:
    """
    Downloads a blob from the specified bucket to the given destination file.

    Args:
        bucket_name (str): The name of the bucket.
        source_blob_name (str): The name of the source blob.
        destination_file_name (str): The name of the destination file.

    Returns:
        bool: True if the blob was successfully downloaded, False otherwise.
    """

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"File {source_blob_name} downloaded in {destination_file_name}.")
    return True


def get_blob_as_string(bucket_name: str, blob_name: str) -> str:
    """
    Retrieves the content of a blob in the specified storage bucket as a string.

    Args:
        bucket_name (str): The name of the storage bucket.
        blob_name (str): The name of the blob to retrieve.

    Returns:
        str: The content of the blob as a string.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob_content = blob.download_as_string()

    print(f"Contenido del blob {blob_name}:")
    print(blob_content.decode("utf-8"))
    return blob_content.decode("utf-8")


def get_blob_as_dataframe(bucket_name: str, blob_name: str) -> pd.DataFrame:
    """
    A function to retrieve a blob from a Google Cloud Storage bucket and convert it into a pandas DataFrame.

    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        blob_name (str): The name of the blob in the specified bucket.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the specified blob.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Descarga el contenido del blob como una cadena de texto
    blob_content = blob.download_as_string()
    csv_string = blob_content.decode("utf-8")

    # Utiliza StringIO para convertir la cadena CSV en un objeto similar a un archivo
    csv_file = StringIO(csv_string)

    # Crea un DataFrame a partir del objeto similar a un archivo
    df = pd.read_csv(csv_file)

    return df


if __name__ == "__main__":
    # example
    blob_name = 'in-vehicle-coupon-recommendation.csv'
    bucket_name = 'ml_lifecycle_airflow'

    # Descargar el archivo del bucket
    df = get_blob_as_dataframe(bucket_name, blob_name)
