#!/usr/bin/env python3

"""
Full-cohort unitig extraction for AMR-Genome-ML.

Primary representation:
    Reference-free unitig presence/absence

Tool:
    unitig-caller

Parameters:
    k = 31
    binary presence/absence downstream

Expected cohort:
    4,227 Klebsiella pneumoniae genome assemblies

This script is intended to run under Linux / WSL2.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


# =========================================================
# Constants
# =========================================================

EXPECTED_N = 4227
DEFAULT_KMER = 31
DEFAULT_HEARTBEAT_MINUTES = 10


# =========================================================
# Helpers
# =========================================================

def timestamp() -> str:
    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def bytes_to_gib(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def write_status(
    path: Path,
    status: str,
    elapsed_hours: float,
    partial_gib: float,
    message: str,
) -> None:

    df = pd.DataFrame(
        [
            {
                "Status": status,
                "Timestamp": timestamp(),
                "Elapsed Hours": elapsed_hours,
                "Partial GiB": partial_gib,
                "Message": message,
            }
        ]
    )

    df.to_csv(
        path,
        index=False,
    )


def append_progress(
    path: Path,
    elapsed_hours: float,
    partial_gib: float,
) -> None:

    exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(
            handle
        )

        if not exists:
            writer.writerow(
                [
                    "Timestamp",
                    "Elapsed Hours",
                    "Partial GiB",
                ]
            )

        writer.writerow(
            [
                timestamp(),
                f"{elapsed_hours:.4f}",
                f"{partial_gib:.4f}",
            ]
        )


def partial_size_gib(
    path: Path
) -> float:

    if not path.exists():
        return 0.0

    return bytes_to_gib(
        path.stat().st_size
    )


def find_executable(
    requested: str
) -> str:

    requested_path = Path(
        requested
    )

    if requested_path.exists():
        return str(
            requested_path.resolve()
        )

    resolved = shutil.which(
        requested
    )

    if resolved is None:
        raise FileNotFoundError(
            f"Executable not found: {requested}"
        )

    return resolved


# =========================================================
# Manifest loading
# =========================================================

def load_manifest(
    manifest_path: Path,
    genome_dir: Path | None,
) -> pd.DataFrame:

    df = pd.read_csv(
        manifest_path,
        dtype={
            "Genome ID": str
        },
    )

    required_columns = {
        "Genome ID",
        "FASTA Path",
    }

    missing_columns = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Manifest is missing columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    df[
        "Genome ID"
    ] = (
        df[
            "Genome ID"
        ]
        .astype(str)
        .str.strip()
    )

    if len(df) != EXPECTED_N:
        raise ValueError(
            f"Expected {EXPECTED_N:,} samples, "
            f"found {len(df):,}."
        )

    if (
        df[
            "Genome ID"
        ]
        .nunique()
        != EXPECTED_N
    ):
        raise ValueError(
            "Duplicate Genome IDs detected."
        )

    # -----------------------------------------------------
    # Optional local remapping
    #
    # If the manifest still contains Colab paths, locate
    # assemblies by filename inside --genome-dir.
    # -----------------------------------------------------

    if genome_dir is not None:

        genome_dir = (
            genome_dir
            .expanduser()
            .resolve()
        )

        if not genome_dir.is_dir():
            raise FileNotFoundError(
                f"Genome directory not found: "
                f"{genome_dir}"
            )

        local_paths = []

        for original in df[
            "FASTA Path"
        ].astype(str):

            filename = Path(
                original
            ).name

            candidate = (
                genome_dir
                / filename
            )

            if not candidate.exists():
                raise FileNotFoundError(
                    "Genome file not found:\n"
                    f"{candidate}"
                )

            local_paths.append(
                str(
                    candidate.resolve()
                )
            )

        df[
            "FASTA Path"
        ] = local_paths

    else:

        df[
            "FASTA Path"
        ] = (
            df[
                "FASTA Path"
            ]
            .astype(str)
        )

    missing_files = [
        path
        for path
        in df[
            "FASTA Path"
        ]
        if not Path(
            path
        ).exists()
    ]

    if missing_files:
        example = "\n".join(
            missing_files[:10]
        )

        raise FileNotFoundError(
            f"{len(missing_files):,} FASTA files "
            f"are missing.\n\n"
            f"Examples:\n{example}\n\n"
            "If the manifest contains old Colab paths, "
            "use --genome-dir."
        )

    compressed_count = sum(
        str(
            path
        ).endswith(
            ".gz"
        )
        for path
        in df[
            "FASTA Path"
        ]
    )

    if compressed_count != EXPECTED_N:
        raise ValueError(
            f"Expected {EXPECTED_N:,} gzip-compressed "
            f"assemblies, found {compressed_count:,}."
        )

    return df


# =========================================================
# Main
# =========================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Run full 4,227-genome unitig extraction."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help=(
            "CSV containing Genome ID and FASTA Path."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Directory for final sequence-variation output."
        ),
    )

    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
        help=(
            "Directory for status, progress and logs."
        ),
    )

    parser.add_argument(
        "--genome-dir",
        type=Path,
        default=None,
        help=(
            "Optional local directory containing all "
            ".fna.gz files. When supplied, old manifest "
            "paths are remapped by filename."
        ),
    )

    parser.add_argument(
        "--unitig-caller",
        default="unitig-caller",
        help=(
            "unitig-caller executable or absolute path."
        ),
    )

    parser.add_argument(
        "--kmer",
        type=int,
        default=DEFAULT_KMER,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=max(
            1,
            os.cpu_count() or 1,
        ),
    )

    parser.add_argument(
        "--heartbeat-minutes",
        type=int,
        default=DEFAULT_HEARTBEAT_MINUTES,
    )

    args = parser.parse_args()


    # =====================================================
    # Platform guard
    # =====================================================

    if os.name != "posix":

        raise RuntimeError(
            "This script must run under Linux/WSL2, "
            "not native Windows Python."
        )


    # =====================================================
    # Resolve tools
    # =====================================================

    unitig_caller = find_executable(
        args.unitig_caller
    )

    gzip_bin = find_executable(
        "gzip"
    )

    time_bin = (
        shutil.which(
            "time"
        )
        or "/usr/bin/time"
    )

    if not Path(
        time_bin
    ).exists():

        raise FileNotFoundError(
            "GNU /usr/bin/time is required."
        )


    # =====================================================
    # Directories
    # =====================================================

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    checkpoint_dir = (
        args.checkpoint_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    work_dir = (
        checkpoint_dir
        / "unitig_full_k31_work"
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # =====================================================
    # Paths
    # =====================================================

    refs_path = (
        work_dir
        / "refs.txt"
    )

    output_prefix = (
        work_dir
        / "unitig_full_k31"
    )

    fifo_path = Path(
        str(
            output_prefix
        )
        + ".pyseer"
    )

    final_output = (
        output_dir
        / "unitig_full_k31.pyseer.gz"
    )

    partial_output = (
        output_dir
        / "unitig_full_k31.pyseer.partial.gz"
    )

    status_path = (
        checkpoint_dir
        / "unitig_full_k31_status.csv"
    )

    progress_path = (
        checkpoint_dir
        / "unitig_full_k31_progress.csv"
    )

    log_path = (
        checkpoint_dir
        / "unitig_full_k31_run.log"
    )

    pid_path = (
        checkpoint_dir
        / "unitig_full_k31_pid.txt"
    )

    summary_path = (
        checkpoint_dir
        / "unitig_full_k31_run_summary.csv"
    )


    # =====================================================
    # Prevent duplicate run
    # =====================================================

    if final_output.exists():

        print(
            "Existing final output detected."
        )

        gzip_test = subprocess.run(
            [
                gzip_bin,
                "-t",
                str(
                    final_output
                ),
            ]
        )

        if (
            gzip_test.returncode == 0
            and final_output.stat().st_size > 0
        ):

            print(
                "PASS: completed output already exists."
            )

            print(
                final_output
            )

            return 0

        raise RuntimeError(
            "A final output exists but failed gzip "
            "integrity validation."
        )


    if pid_path.exists():

        try:

            old_pid = int(
                pid_path
                .read_text()
                .strip()
            )

            if process_exists(
                old_pid
            ):

                raise RuntimeError(
                    "Another unitig job appears to be "
                    f"running with PID {old_pid}."
                )

        except ValueError:
            pass

        pid_path.unlink(
            missing_ok=True
        )


    # =====================================================
    # Archive stale partial output
    # =====================================================

    if partial_output.exists():

        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        archived = (
            checkpoint_dir
            / (
                "unitig_full_k31_interrupted_"
                f"{stamp}.partial.gz"
            )
        )

        shutil.move(
            partial_output,
            archived,
        )

        print(
            "Archived stale partial output:"
        )

        print(
            archived
        )


    # =====================================================
    # Load and validate cohort
    # =====================================================

    manifest_path = (
        args.manifest
        .expanduser()
        .resolve()
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    df = load_manifest(
        manifest_path=manifest_path,
        genome_dir=args.genome_dir,
    )


    # =====================================================
    # Write local refs.txt
    # =====================================================

    with refs_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for path in df[
            "FASTA Path"
        ]:

            handle.write(
                str(
                    Path(
                        path
                    ).resolve()
                )
                + "\n"
            )


    # =====================================================
    # Run metadata
    # =====================================================

    print("=" * 70)
    print("FULL 4,227-GENOME UNITIG EXTRACTION")
    print("=" * 70)

    print(
        f"Platform       : "
        f"{platform.platform()}"
    )

    print(
        f"Samples        : "
        f"{len(df):,}"
    )

    print(
        f"k-mer size     : "
        f"{args.kmer}"
    )

    print(
        f"Threads        : "
        f"{args.threads}"
    )

    print(
        f"unitig-caller  : "
        f"{unitig_caller}"
    )

    print(
        f"Manifest       : "
        f"{manifest_path}"
    )

    print(
        f"Final output   : "
        f"{final_output}"
    )

    print(
        f"Run log        : "
        f"{log_path}"
    )

    print()


    # =====================================================
    # FIFO
    # =====================================================

    fifo_path.unlink(
        missing_ok=True
    )

    os.mkfifo(
        fifo_path
    )


    # =====================================================
    # Start streaming gzip
    # =====================================================

    partial_handle = partial_output.open(
        "wb"
    )

    fifo_reader = fifo_path.open(
        "rb",
        buffering=0,
    )

    gzip_process = subprocess.Popen(
        [
            gzip_bin,
            "-1",
            "-c",
        ],
        stdin=fifo_reader,
        stdout=partial_handle,
        stderr=subprocess.PIPE,
    )


    # =====================================================
    # unitig-caller command
    # =====================================================

    command = [
        time_bin,
        "-v",
        unitig_caller,
        "--call",
        "--refs",
        str(
            refs_path
        ),
        "--out",
        str(
            output_prefix
        ),
        "--kmer",
        str(
            args.kmer
        ),
        "--threads",
        str(
            args.threads
        ),
        "--pyseer",
    ]


    print(
        "Command:"
    )

    print(
        " ".join(
            shlex.quote(
                x
            )
            for x in command
        )
    )

    print()


    # =====================================================
    # Status initialization
    # =====================================================

    start_time = time.time()

    pid_path.write_text(
        str(
            os.getpid()
        )
    )

    progress_path.unlink(
        missing_ok=True
    )

    write_status(
        status_path,
        status="STARTING",
        elapsed_hours=0.0,
        partial_gib=0.0,
        message=(
            "Launching full-cohort unitig extraction"
        ),
    )


    unitig_process = None


    # =====================================================
    # Signal cleanup
    # =====================================================

    interrupted = False


    def stop_children(
        *_args
    ):

        nonlocal interrupted

        interrupted = True

        print()
        print(
            "Termination requested. "
            "Stopping child processes..."
        )

        if (
            unitig_process is not None
            and
            unitig_process.poll() is None
        ):
            unitig_process.terminate()

        if gzip_process.poll() is None:
            gzip_process.terminate()


    signal.signal(
        signal.SIGINT,
        stop_children,
    )

    signal.signal(
        signal.SIGTERM,
        stop_children,
    )


    # =====================================================
    # Run
    # =====================================================

    try:

        with log_path.open(
            "w",
            encoding="utf-8",
        ) as log_handle:

            unitig_process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )


            write_status(
                status_path,
                status="RUNNING",
                elapsed_hours=0.0,
                partial_gib=0.0,
                message=(
                    "Full cohort graph construction "
                    "and unitig calling"
                ),
            )


            next_heartbeat = (
                time.time()
                +
                args.heartbeat_minutes
                * 60
            )


            while (
                unitig_process.poll()
                is None
            ):

                time.sleep(
                    10
                )


                if interrupted:
                    break


                if (
                    gzip_process.poll()
                    is not None
                ):

                    unitig_process.terminate()

                    raise RuntimeError(
                        "gzip process terminated "
                        "unexpectedly."
                    )


                if (
                    time.time()
                    >= next_heartbeat
                ):

                    elapsed_hours = (
                        time.time()
                        - start_time
                    ) / 3600


                    current_gib = (
                        partial_size_gib(
                            partial_output
                        )
                    )


                    append_progress(
                        progress_path,
                        elapsed_hours,
                        current_gib,
                    )


                    write_status(
                        status_path,
                        status="RUNNING",
                        elapsed_hours=elapsed_hours,
                        partial_gib=current_gib,
                        message="Heartbeat",
                    )


                    print(
                        f"[{timestamp()}] "
                        f"{elapsed_hours:.2f} h | "
                        f"partial "
                        f"{current_gib:.3f} GiB"
                    )


                    next_heartbeat += (
                        args.heartbeat_minutes
                        * 60
                    )


            if interrupted:

                if (
                    unitig_process.poll()
                    is None
                ):
                    unitig_process.terminate()

                gzip_process.terminate()

                raise KeyboardInterrupt


            unitig_returncode = (
                unitig_process.wait()
            )


        # =================================================
        # Finish gzip
        # =================================================

        fifo_reader.close()

        gzip_returncode = (
            gzip_process.wait()
        )

        partial_handle.close()


        if unitig_returncode != 0:

            raise RuntimeError(
                "unitig-caller failed with "
                f"exit code {unitig_returncode}. "
                f"See log:\n{log_path}"
            )


        if gzip_returncode != 0:

            gzip_stderr = (
                gzip_process.stderr.read()
                .decode(
                    errors="replace"
                )
                if gzip_process.stderr
                else ""
            )

            raise RuntimeError(
                "gzip failed with "
                f"exit code {gzip_returncode}.\n"
                f"{gzip_stderr}"
            )


        # =================================================
        # Validate gzip
        # =================================================

        write_status(
            status_path,
            status="VALIDATING",
            elapsed_hours=(
                time.time()
                - start_time
            ) / 3600,
            partial_gib=partial_size_gib(
                partial_output
            ),
            message="Checking gzip integrity",
        )


        print()
        print(
            "Validating gzip output..."
        )


        gzip_test = subprocess.run(
            [
                gzip_bin,
                "-t",
                str(
                    partial_output
                ),
            ]
        )


        if gzip_test.returncode != 0:

            raise RuntimeError(
                "gzip integrity check failed."
            )


        # =================================================
        # Promote partial -> final
        # =================================================

        os.replace(
            partial_output,
            final_output,
        )


        elapsed_hours = (
            time.time()
            - start_time
        ) / 3600


        final_gib = bytes_to_gib(
            final_output.stat().st_size
        )


        # =================================================
        # Final summary
        # =================================================

        summary = pd.DataFrame(
            [
                {
                    "Status": "COMPLETE",
                    "Genomes": EXPECTED_N,
                    "k-mer Size": args.kmer,
                    "Threads": args.threads,
                    "Start Time": datetime.fromtimestamp(
                        start_time
                    ).astimezone().isoformat(
                        timespec="seconds"
                    ),
                    "Stop Time": timestamp(),
                    "Wall Hours": elapsed_hours,
                    "Compressed Output GiB": final_gib,
                    "Phenotype Used": False,
                    "Manifest": str(
                        manifest_path
                    ),
                    "Output": str(
                        final_output
                    ),
                    "Log": str(
                        log_path
                    ),
                }
            ]
        )


        summary.to_csv(
            summary_path,
            index=False,
        )


        write_status(
            status_path,
            status="COMPLETE",
            elapsed_hours=elapsed_hours,
            partial_gib=final_gib,
            message=(
                "Full cohort unitig extraction complete"
            ),
        )


        pid_path.unlink(
            missing_ok=True
        )

        fifo_path.unlink(
            missing_ok=True
        )


        print()
        print("=" * 70)
        print("FULL UNITIG EXTRACTION COMPLETE")
        print("=" * 70)

        print(
            f"Wall runtime      : "
            f"{elapsed_hours:.2f} h"
        )

        print(
            f"Final output size : "
            f"{final_gib:.3f} GiB"
        )

        print(
            f"Final output      : "
            f"{final_output}"
        )

        print(
            f"Run summary       : "
            f"{summary_path}"
        )

        print()

        print(
            "PASS: full 4,227-genome unitig "
            "extraction completed successfully."
        )

        return 0


    # =====================================================
    # Interrupt
    # =====================================================

    except KeyboardInterrupt:

        elapsed_hours = (
            time.time()
            - start_time
        ) / 3600


        current_gib = partial_size_gib(
            partial_output
        )


        write_status(
            status_path,
            status="INTERRUPTED",
            elapsed_hours=elapsed_hours,
            partial_gib=current_gib,
            message=(
                "Run interrupted before completion"
            ),
        )


        print()
        print(
            "Run interrupted."
        )

        print(
            "Partial output retained for audit only:"
        )

        print(
            partial_output
        )

        return 130


    # =====================================================
    # Failure
    # =====================================================

    except Exception as exc:

        elapsed_hours = (
            time.time()
            - start_time
        ) / 3600


        current_gib = partial_size_gib(
            partial_output
        )


        write_status(
            status_path,
            status="FAILED",
            elapsed_hours=elapsed_hours,
            partial_gib=current_gib,
            message=str(
                exc
            ).replace(
                ",",
                ";"
            ),
        )


        print()
        print(
            "FAILED:"
        )

        print(
            exc
        )

        return 1


    finally:

        pid_path.unlink(
            missing_ok=True
        )

        fifo_path.unlink(
            missing_ok=True
        )

        try:
            fifo_reader.close()
        except Exception:
            pass

        try:
            partial_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(
        main()
    )