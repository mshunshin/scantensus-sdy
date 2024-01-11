from dataclasses import dataclass
import os

from minio import Minio
import io 

from typing import Union

@dataclass
class MinioConfig:
    """
    Represents standard MinIO configuration informatino.
    """
    
    endpoint: str
    bucket: str
    
    access_key: str
    secret_key: str


    def build_client(self) -> Minio:
        """
        Builds a MinIO client from the given configuration.
        """
        return Minio(endpoint=self.endpoint, access_key=self.access_key, secret_key=self.secret_key)

    def write_to_minio(self, data: Union[str, bytes, io.BytesIO], file: str, content_type: str = "application/octet-stream"):
        client = self.build_client()
        
        data_io = io.BytesIO()

        if type(data) is str:
            data_io.write(data.encode())
        elif type(data) is bytes:
            data_io.write(data)
        elif type(data) is io.BytesIO:
            data_io = data
            data_io.seek(0,2)
        else:
            raise Exception("Data is not str / bytes / io.BytesIO")

        data_io_len = data_io.tell()
        data_io.seek(0)
        client.put_object(bucket_name=self.bucket_name,
                                     object_name=file,
                                     data=data_io,
                                     length=data_io_len,
                                     content_type=content_type)


    @staticmethod
    def from_env() -> 'MinioConfig':
        """
        Loads a MinIO configuration from the environment.
        """
        
        return MinioConfig(
            endpoint=os.getenv('MINIO_ENDPOINT'),
            bucket=os.getenv('MINIO_BUCKET'),
            access_key=os.getenv('BUCKET_ACCESS_KEY'),
            secret_key=os.getenv('BUCKET_ACCESS_KEY_SECRET')
        )