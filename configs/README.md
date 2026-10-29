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

Enemy manager profiles: baseline MD5 `b0cc28ec904e9efce2439ad857c938b5` uses RVA
`0xAA1E90`; MD5 `fe47d9c26828e4671a2d2e4bf00af5a4` uses `0xAEE0B0`.
The inline pool starts at manager + 8, with 256 slots of stride `0x10B0`.
Slot fields: x/y float32 at `0xB0/0xB4`, signed HP int32 at `0x234`, flags bytes
at `0xBC/0xBD/0xBE`. Sources: reference `tools/read_enemies.py`, `tools/offsets.py`,
`tools/addr_cache.json`, and MEMMAP section 6. These remain unverified locally.
Custom profile files must now include `EM_RVA`; missing entries fail explicitly.
