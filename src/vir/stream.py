#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:12:14 2026

@author: pvargas21
"""


from concurrent.futures import ThreadPoolExecutor
import numpy as np


def stream_engine(
    n_total,
    chunk_size,
    buffer_factory,
    read_chunk,
    process,
    write_chunk,
    *,
    buffer_view=None,
    progress=None,
):
    """
    Three-buffer streaming pipeline.

    One reader thread, one writer thread, main-thread processing.

    buffer_factory(chunk_size) -> reusable buffer
    buffer_view(buffer, n)     -> view for a chunk of n elements
    read_chunk(buffer, start, end)
    process(buffer, start, end)
    write_chunk(result, start, end)

    process may modify its input in place and return None,
    or return a replacement array.

    A returned replacement must remain valid until its write
    completes. Do not retain references to reusable buffers.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if n_total < 0:
        raise ValueError("n_total must be nonnegative")

    if buffer_view is None:
        buffer_view = lambda b, n: b[:n]

    n_chunks = (n_total + chunk_size - 1) // chunk_size

    if n_chunks == 0:
        return

    buffers = [buffer_factory(chunk_size) for _ in range(3)]

    read_futures = {}
    write_futures = [None, None, None]

    def submit_read(k, reader):
        slot = k % 3

        # This buffer may still be owned by an earlier write.
        wf = write_futures[slot]
        if wf is not None:
            wf.result()
            write_futures[slot] = None

        start = k * chunk_size
        end = min(start + chunk_size, n_total)

        b = buffer_view(buffers[slot], end - start)

        read_futures[k] = reader.submit(
            read_chunk, b, start, end
        )

    with ThreadPoolExecutor(1) as reader, \
         ThreadPoolExecutor(1) as writer:

        # Initial prefetch.
        for k in range(min(3, n_chunks)):
            submit_read(k, reader)

        for k in range(n_chunks):

            slot = k % 3
            start = k * chunk_size
            end = min(start + chunk_size, n_total)

            # Wait for this chunk's read.
            read_futures.pop(k).result()

            # Refill the next available buffer before processing.
            # This overlaps the next read with current processing.
            next_k = k + 1

            if next_k < n_chunks and next_k >= 3:
                submit_read(next_k, reader)

            b = buffer_view(buffers[slot], end - start)

            # Process.
            result = process(b, start, end)
            if result is None:
                result = b

            # Submit asynchronous write.
            write_futures[slot] = writer.submit(
                write_chunk, result, start, end
            )

            if progress is not None:
                progress(start, end, n_total)

        # Drain all outstanding writes.
        for future in write_futures:
            if future is not None:
                future.result()



from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import h5py
import numpy as np
import numexpr as ne


def normalize_sample_vol(infile, outfile=None, chunk_size=8):

    infile = Path(infile)

    if outfile is None:
        outdir = infile.parent / "vols"
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{infile.stem}_vol_norm.npy"

    outfile = Path(outfile)

    with h5py.File(infile, 'r') as f:

        dset = f['exchange/data']
        na, nv, nu = dset.shape

        w0, w1 = white_fields(f)
        dw = (w1 - w0) / np.float32(na)

        # Allocate three full-sized buffers.
        def buffer_factory(n):
            return np.empty(
                (n, nv, nu), dtype=np.float32
            )

        # HDF5 reader.
        def read_chunk(buf, start, end):
            dset.read_direct(
                buf,
                source_sel=np.s_[start:end]
            )

        # In-place normalization.
        def process(batch, start, end):

            alpha = (
                np.arange(start, end, dtype=np.float32)
                / np.float32(na - 1)
            )[:, None, None]

            ne.evaluate(
                "-log(b / (w0 + alpha * dw))",
                local_dict={
                    'b': batch,
                    'w0': w0,
                    'dw': dw,
                    'alpha': alpha,
                },
                out=batch,
                casting='same_kind',
            )

        with open(outfile, 'wb') as out:

            np.lib.format.write_array_header_1_0(
                out,
                {
                    'descr': '<f4',
                    'fortran_order': False,
                    'shape': (na, nv, nu),
                }
            )

            # Sequential output writer.
            def write_chunk(batch, start, end):
                batch.tofile(out)

            def report(start, end, total):
                print(
                    f"Progress: {100 * end / total:.1f}%",
                    end='\r',
                    flush=True,
                )

            stream_engine(
                n_total=na,
                chunk_size=chunk_size,
                buffer_factory=buffer_factory,
                read_chunk=read_chunk,
                process=process,
                write_chunk=write_chunk,
                progress=report,
            )

    print()
    return outfile



from pathlib import Path
import numpy as np


def process_npy_slabs(
    infile,
    process,
    outfile=None,
    chunk_rows=4,
    progress=None,
):

    infile = Path(infile)

    m = np.load(infile, mmap_mode='r')

    assert m.dtype == np.float32
    assert m.flags.c_contiguous
    assert m.ndim == 3

    na, nv, nu = m.shape
    in_off = m.offset

    del m

    row_bytes = nu * 4

    if outfile is None:
        out_path = infile
        out_off = in_off

    else:
        out_path = Path(outfile)

        with open(out_path, 'wb') as f:
            np.lib.format.write_array_header_1_0(
                f,
                {
                    'descr': '<f4',
                    'fortran_order': False,
                    'shape': (na, nv, nu),
                }
            )

            out_off = f.tell()
            f.truncate(out_off + na * nv * row_bytes)

    # Three slab buffers.
    def buffer_factory(n):
        return np.empty(
            (na, n, nu),
            dtype=np.float32,
        )

    def buffer_view(buf, n):
        return buf[:, :n, :]

    # Keep independent file handles open.
    with open(infile, 'rb', buffering=0) as rf, \
         open(out_path, 'r+b', buffering=0) as wf:

        def read_chunk(buf, j0, j1):

            for a in range(na):

                rf.seek(
                    in_off + (a * nv + j0) * row_bytes
                )

                row_view = memoryview(buf[a]).cast('B')

                got = rf.readinto(row_view)

                if got != buf[a].nbytes:
                    raise IOError("Short read")

        def write_chunk(buf, j0, j1):

            for a in range(na):

                wf.seek(
                    out_off + (a * nv + j0) * row_bytes
                )

                wf.write(memoryview(buf[a]).cast('B'))

        def report(start, end, total):
            if progress:
                progress(start, end, total)

        stream_engine(
            n_total=nv,
            chunk_size=chunk_rows,
            buffer_factory=buffer_factory,
            buffer_view=buffer_view,
            read_chunk=read_chunk,
            process=process,
            write_chunk=write_chunk,
            progress=report,
        )
