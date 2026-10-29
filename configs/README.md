# Configuration

`memory_profiles.json` maps executable MD5 hashes to module-relative addresses.
Both included profiles come from `Reference/th06nc-autododge-main`: the baseline
`offsets.py` and the newer `addr_cache.json` entry. They are not locally validated.
Unknown or incomplete profiles raise `UnsupportedBuildError`.

The fixed player and bullet structure offsets currently live in
`env/memory_reader.py`. A matching hash alone does not verify their interpretation.
Check player coordinates, bullet counts, deaths, and failure menus in the game
before enabling input. Never guess an address to bypass a version mismatch.

The upstream address cache contains mixed confidence levels. Keep that original
cache as the source for per-address evidence. `REFERENCE_LICENSE.txt` preserves
the upstream MIT notice for the extracted layout/profile data.
