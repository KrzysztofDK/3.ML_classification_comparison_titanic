"""Module to load data."""

import sys

import pandas as pd
import chardet

from src.logger import logging
from src.exception import CustomException


def load_csv_with_detected_encoding(path: str) -> pd.DataFrame:
    """
    Detect the encoding of the given csv file and write that file to DataFrame.

    Args:
        path (str): Path to CSV file to check encoding and confidence and write.

    Returns:
        pd.DataFrame object.
    """

    logging.info("CSV file loading started.")

    try:
        with open(path, "rb") as f:
            sample = f.read(10000)
            result = chardet.detect(sample)
            encoding = result["encoding"]
            confidance = result["confidence"]

            if encoding is None or result["confidence"] < 0.8:
                print(
                    "Note: Low confidence in detected encoding. Below 0.8. or encoding is None. "
                    "Process will be continued."
                )
                logging.info(
                    "Low confidence in detected encoding. Below 0.8. or encoding is None."
                )

            print(
                f"Detected encoding: {encoding} (confidence: {confidance:.2f}). "
                "Dataframe created."
            )

            df = pd.read_csv(path, encoding=encoding, low_memory=False)

            logging.info("Dataframe from CSV file created.")

            return df

    except Exception as e:
        logging.info("Function to load data has encountered a problem.")
        raise CustomException(e, sys) from e
