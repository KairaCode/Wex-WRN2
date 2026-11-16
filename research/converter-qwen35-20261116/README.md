# The converter on a whole real checkpoint: Qwen3.5-35B-A3B MXFP4 (2026-11-16)

The first run of [`wex4/wex4_convert.py`](../../wex4/wex4_convert.py) on a whole published checkpoint:
`pack --experts raw --names qwen38`, then `wex4.py verify`, then `unpack`, on every shard of
`amd/Qwen3.5-35B-A3B-MXFP4`. It answers the "most wanted" item of section 5 of
[docs/HELP-WANTED.md](../../docs/HELP-WANTED.md) for one profile and one checkpoint.

## Question

Does `pack` turn every routed expert of a real MXFP4 checkpoint into an expert record under an existing
naming profile, do both checks of the container pass, and does `unpack` give every source file back with
its SHA-256?

## Setup

- **Checkpoint:** `amd/Qwen3.5-35B-A3B-MXFP4` at commit `2e19c657` of its Hugging Face repository, all 14
  safetensors shards, 24,600,620,848 bytes in all, downloaded whole. Its routed experts carry the names
  of the `qwen38` profile (`pack` matched every one of them, see below).
- **Code:** `wex4/wex4_convert.py`, `wex4/wex4.py` and the library `wex4/out/wex4_tools.dll` of this
  repository on the morning of 2026-11-16; the log does not name the commit. The library was built with
  MSYS2 UCRT64 gcc 16.2.0.
- **Machine:** Intel Core i7-7700K (4 cores, 8 threads), 32 GB of RAM, Windows 10, MSYS2 UCRT64 Python.
  The checkpoint and the container were on drive H: of that machine. **Two downloads were running during
  the whole run**, on the same machine and the same drive, so the times below are upper bounds, not
  speeds of the converter.
- **Commands** (paths shortened):

  ```sh
  python -I wex4/wex4_convert.py pack OUT.wex4 <MODEL>/*.safetensors --experts raw --names qwen38
  python -I wex4/wex4.py verify OUT.wex4
  python -I wex4/wex4_convert.py unpack OUT.wex4 <UNPACKED>
  # then the SHA-256 of each of the 14 unpacked files against its source
  ```

## Results

| Step | What it printed | Time |
|---|---|---:|
| `pack --experts raw --names qwen38` | `10240 expert records; 0 experts kept as ordinary tensors (float or FP8, not MXFP4)`: 40 layers x 256 experts, every routed expert an expert record; root `44aae6a17c4cae7ddc5f02dfc7c94f10d1e257bc6385b7c90feb28d4291ca5c5` | 607 s |
| `wex4.py verify` | `WEX4 4.0`, the same root; file 22,070,321,152 B, index 20,551,095 B, 0 blobs, 11,543 records, 63,171 tensors; decoded 24,591,905,888 B, stored 22,049,763,328 B, ratio 0.8966; codecs: 10,240 RAW (the expert records) and 1,303 WBP1 (the other tensors); content digest `2aac93c8...dcdbd1069c`; "every hash checked, every record decoded" | 257 s |
| `unpack`, then SHA-256 of every file | 14 of 14 files with the SHA-256 of their source, 0 different | 450 s |

The expert records are RAW and store their bytes as they are, so the ratio 0.8966 comes from the WBP1
records that hold the other tensors (1,303 records). A WNR3 pack (`--experts nr3`) of this checkpoint
was not run.

## What it shows, and what it does not

- The `qwen38` profile fits this export as it is: no `incomplete expert`, no expert left as ordinary
  tensors, and the round trip is byte for byte on the whole checkpoint.
- One checkpoint, one profile, RAW expert records, one run. The other fifteen profiles have still met no
  whole checkpoint. No engine read the container (no engine of this repository runs Qwen3.5).
- The times are of a machine with two downloads running on the same drive; they are not a speed of the
  converter.

## Raw files

| File | What it is |
|---|---|
| `pack-verify-unpack.log` | the log of the run, as the run's script wrote it. Two edits: the path of the container is shortened to `<WORK>/`, and the owner and group of the `ls -l` line are replaced by `<user>` and `<group>`. Its first two lines come from the build of `wex4/out/wex4_tools.dll` before the pack. The unpack lines name only the last three shards; the line after them gives the count for all 14. |
