# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import json
import random
import string
import subprocess
import sys
import os

from typing import *
from pathlib import Path

import psycopg2
import requests

from google.cloud import bigquery
from google.oauth2 import service_account


IS_IN_CI = os.getenv("IS_IN_CI", "0") == "1"


def eprint(*args):
    print(*args, file=sys.stderr, flush=True)


_bigquery_client_and_config = None


def bigquery_client_and_config(
    key_path: str = "gcp_auth.json", schema: Optional[List[bigquery.SchemaField]] = None
):
    if not Path(key_path).exists():
        raise RuntimeError(f"{key_path} was not found, did you forget to mount it?")

    global _bigquery_client_and_config
    if _bigquery_client_and_config is None:
        credentials = service_account.Credentials.from_service_account_file(
            key_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = bigquery.Client(
            credentials=credentials,
            project=credentials.project_id,
        )

        if schema is None:
            schema = [
                bigquery.SchemaField("r", "STRING", mode="REQUIRED"),
            ]

        job_config = bigquery.LoadJobConfig(
            schema=schema,
        )
        job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
        job_config.autodetect = True
        _bigquery_client_and_config = (client, job_config)

    return _bigquery_client_and_config


def bigquery_upload(jsonl_file: Path, dataset_id: str, table_id: str) -> int:
    client, job_config = bigquery_client_and_config()
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    with open(jsonl_file, "rb") as source_file:
        job = client.load_table_from_file(
            source_file,
            table_ref,
            location="us-west1",  # Must match the destination dataset location.
            job_config=job_config,
        )  # API request

    job.result()

    return job


def query_ec2_metadata(key: str) -> str:
    url = f"http://169.254.169.254/latest/meta-data/{key}"
    response = requests.get(url)
    return response.content.decode().strip()


def postgres_upload(jsonl_file: Path, database: str, table_name: str) -> int:
    """
    Uploads records in jsonl_file (one JSON document per line) to postgres
    """
    rows = [(json.dumps(d["r"]),) for d in load_jsonl(jsonl_file=jsonl_file)]
    sql = f"INSERT INTO {table_name} (r) VALUES (%s)"
    conn = None
    password = os.environ["POSTGRES_PASSWORD"]
    ip = os.environ["POSTGRES_IP"]
    user = os.getenv("POSTGRES_USER", "ci")
    try:
        conn = psycopg2.connect(
            host=ip,
            database=database,
            user=user,
            password=password,
        )
        cur = conn.cursor()
        cur.executemany(sql, rows)
        conn.commit()
        cur.close()
    finally:
        if conn is not None:
            conn.close()

    return len(rows)


def aws_download(blob_name: str, out_path: Path, bucket_name: str = "scorecard-models"):
    command = ["aws", "s3", "cp", f"s3://{bucket_name}/{blob_name}", out_path]
    if IS_IN_CI:
        command.append("--no-progress")
    command = [str(c) for c in command]
    eprint(f"+ {' '.join(command)}")
    subprocess.run(command, check=True, stdout=sys.stderr)
    return out_path
