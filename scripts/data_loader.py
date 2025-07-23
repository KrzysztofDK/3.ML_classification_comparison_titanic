import pandas as pd
import chardet

def load_csv_with_detected_encoding(path: str) -> pd.DataFrame:
    """
    Detect the encoding of the given csv file and write that file to DataFrame.

    Args:
        path (str): Path to CSV file to examine and write.

    Raises:
        FileNotFoundError: File not found.
        UnicodeDecodeError: Unicode error with detected encoding.
        RuntimeError: Failed to load CSV file.

    Returns:
        df: DataFrame object.
    """
    try:
        with open(path, 'rb') as f:
            sample = f.read(10000)
            result = chardet.detect(sample)
            encoding = result['encoding']
            confidance = result['confidence']
            if encoding is None or result['confidence'] < 0.8:
                print("Note: Low confidence in detecting encoding. Below 0.8.")
            print(f'[INFO] Detected encoding: {encoding} (confidence: {confidance:.2f})')
            
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
            return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f'[ERROR] File not found at path: {path}')
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(f'[ERROR] Unicode error with detected encoding: {e}')
    except Exception as e:
        raise RuntimeError(f'[ERROR] Failed to load CSV: {e}')