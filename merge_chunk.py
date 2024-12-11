import os
import pandas as pd
from pathlib import Path
import shutil
import json
import argparse
import time
from datetime import datetime, timedelta
import psutil

def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    return f"{memory_gb:.2f}GB"

def format_time(seconds):
    """Format seconds into human readable time"""
    return str(timedelta(seconds=round(seconds)))

def validate_chunk_file(filepath):
    """Validate if a chunk file is readable and not corrupted"""
    try:
        df = pd.read_csv(filepath, compression='gzip', nrows=1)
        return True
    except Exception as e:
        log(f"Found corrupted chunk file {filepath}: {str(e)}")
        return False

def merge_chunks(chunks_dir, final_dir, year):
    """Merge all chunks into HDF5 file with progress tracking"""
    try:
        start_time = time.time()
        log(f"Starting merge process for year {year}")
        log(f"Initial memory usage: {get_memory_usage()}")

        # Setup directories
        chunks_dir = Path(chunks_dir)
        final_dir = Path(final_dir)
        corrupt_dir = chunks_dir / "corrupted"
        corrupt_dir.mkdir(exist_ok=True)

        # Get and validate chunks
        chunks = sorted(chunks_dir.glob("chunk_*.csv.gz"))
        total_chunks = len(chunks)
        
        if not total_chunks:
            log("No chunks found to merge")
            return

        log(f"Found {total_chunks} chunks to process")
        
        # Validate chunks first
        log("Validating chunks...")
        valid_chunks = []
        for i, chunk in enumerate(chunks, 1):
            if validate_chunk_file(chunk):
                valid_chunks.append(chunk)
            else:
                try:
                    shutil.move(str(chunk), str(corrupt_dir / chunk.name))
                except Exception as e:
                    log(f"Error moving corrupted file {chunk}: {str(e)}")
            
            if i % 100 == 0:
                log(f"Validated {i}/{total_chunks} chunks ({(i/total_chunks)*100:.1f}%)")

        total_valid = len(valid_chunks)
        log(f"Found {total_valid} valid chunks to merge")

        # Start merging
        total_records = 0
        final_file = final_dir / f"cdl_california_{year}.h5"
        
        with pd.HDFStore(final_file, mode='w') as store:
            for i, chunk_file in enumerate(valid_chunks, 1):
                try:
                    chunk_start = time.time()
                    df = pd.read_csv(chunk_file, compression='gzip')
                    
                    # Append with min_itemsize for string columns
                    store.append('data', 
                               df, 
                               index=False,
                               min_itemsize={'crop_name': 40},  # Allow longer strings
                               data_columns=True)  # Enable indexing
                    
                    total_records += len(df)
                    
                    # Calculate progress and timing
                    elapsed = time.time() - start_time
                    avg_time_per_chunk = elapsed / i
                    remaining_chunks = total_valid - i
                    estimated_remaining = avg_time_per_chunk * remaining_chunks
                    
                    log(f"Progress: {i}/{total_valid} chunks "
                        f"({(i/total_valid)*100:.1f}%) - "
                        f"Records: {total_records:,} - "
                        f"Memory: {get_memory_usage()} - "
                        f"Est. remaining: {format_time(estimated_remaining)}")
                    
                    # Clear memory
                    del df
                    
                except Exception as e:
                    log(f"Error processing {chunk_file.name}: {str(e)}")
                    try:
                        shutil.move(str(chunk_file), str(corrupt_dir / chunk_file.name))
                    except Exception as move_error:
                        log(f"Error moving corrupted file: {str(move_error)}")
                    continue

            # Store metadata
            store.get_storer('data').attrs.metadata = {
                'total_records': total_records,
                'year': year,
                'processing_date': datetime.now().isoformat(),
                'num_chunks_processed': total_valid
            }

        # Verify final file
        log("Verifying final file...")
        verification_successful = False
        try:
            with pd.HDFStore(final_file, mode='r') as store:
                verify_count = store.get_storer('data').nrows
                if verify_count == total_records:
                    log(f"Verification successful: {verify_count:,} records")
                    verification_successful = True
                else:
                    log(f"Verification failed: Expected {total_records:,} but found {verify_count:,}")
        except Exception as e:
            log(f"Error verifying final file: {str(e)}")

        # Clean up only if verification was successful
        if verification_successful:
            try:
                if chunks_dir.exists():
                    shutil.rmtree(chunks_dir)
                    log("Cleaned up chunk files")
            except Exception as e:
                log(f"Error during cleanup: {str(e)}")
                log("Keeping chunk files for safety")
        else:
            log("Keeping chunk files due to verification failure")

        # Final summary
        end_time = time.time()
        total_time = end_time - start_time
        
        log("\nMerge Process Summary:")
        log(f"Total time: {format_time(total_time)}")
        log(f"Total records: {total_records:,}")
        log(f"Processing speed: {total_records/total_time:.0f} records/second")
        log(f"Final memory usage: {get_memory_usage()}")
        log("Merge process completed successfully")

    except Exception as e:
        log(f"Critical error during merge process: {str(e)}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Merge CDL chunks for a specific year")
    parser.add_argument("year", type=int, help="Year to merge (e.g., 2008)")
    parser.add_argument("--input", type=str, default="/Volumes/aunsh_hd/cdl_processed",
                        help="Input directory path containing chunks")
    parser.add_argument("--output", type=str, default="/Volumes/aunsh_hd/cdl_processed",
                        help="Output directory path for final file")

    args = parser.parse_args()

    chunks_dir = Path(args.input) / "chunks" / str(args.year)
    final_dir = Path(args.output) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    merge_chunks(chunks_dir, final_dir, args.year)

if __name__ == "__main__":
    main()
