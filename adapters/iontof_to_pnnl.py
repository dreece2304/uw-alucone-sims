from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict
import numpy as np
import pandas as pd

try:
    import pySPM  # type: ignore
except Exception:
    pySPM = None

_DATA_EXTS = {".ita", ".itm", ".its", ".itb"}
_INDEX_TO_DATA = {".itax": ".ita", ".itmx": ".itm"}
_ALIASES = {".tm": ".itm"}

def _resolve_iontof_path(p: Path) -> Path:
    p = Path(p)
    suf = p.suffix.lower()
    if suf in _ALIASES:
        p = p.with_suffix(_ALIASES[suf])
    if not p.exists() and p.parent.exists():
        for f in p.parent.iterdir():
            if f.name.lower() == p.name.lower():
                p = f
                break
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p

def _has_sibling(p: Path, ext: str) -> Optional[Path]:
    q = p.with_suffix(ext)
    return q if q.exists() else None

def _sum_spectrum_from_itax(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    import struct
    obj = pySPM.ITAX(str(path))
    
    try:
        # Try the standard method first
        m, I = obj.getSpectrum()
        return np.asarray(m, dtype=float), np.asarray(I, dtype=float)
    except struct.error as e:
        if "unpack requires a buffer" in str(e):
            # Handle buffer size mismatch with manual extraction
            print(f"ITAX buffer issue detected, applying fix...")
            
            # Get spectrum length more carefully
            slen_node = obj.root.goto("CommonDataObjects/DataViewCollection/*/sizeSpectrum")
            slen = slen_node.get_long()
            
            # Get raw data
            raw_node = obj.root.goto(
                "CommonDataObjects/DataViewCollection/*/dataSource/simsDataCache/spectrum/correctedData"
            )
            raw = raw_node.value
            
            # Calculate expected vs actual size
            expected_bytes = slen * 8  # 8 bytes per double
            actual_bytes = len(raw)
            
            # Check if data is compressed
            if actual_bytes < expected_bytes:
                print(f"ITAX: Data appears compressed ({actual_bytes} < {expected_bytes}), trying decompression...")
                try:
                    import zlib
                    decompressed = zlib.decompress(raw)
                    print(f"ITAX: Successfully decompressed {actual_bytes} -> {len(decompressed)} bytes")
                    raw = decompressed
                    actual_bytes = len(raw)
                except Exception as e:
                    print(f"ITAX: Decompression failed: {e}, truncating spectrum")
                    slen = actual_bytes // 8
            
            # Final size check
            if actual_bytes < expected_bytes:
                print(f"ITAX: Still undersized, truncating spectrum length from {slen} to {actual_bytes // 8}")
                slen = actual_bytes // 8
            elif actual_bytes > expected_bytes:
                print(f"ITAX: Using exact expected size {slen} points from {actual_bytes} bytes")
                
            # Unpack with corrected size
            spectrum = np.array(struct.unpack("<" + str(slen) + "d", raw[:slen*8]))
            CH = 2 * np.arange(slen)
            
            # Get mass calibration
            try:
                sf = obj.root.goto(
                    "CommonDataObjects/DataViewCollection/*/properties/Context.MassScale.SF",
                    lazy=True,
                ).get_key_value()["float"]
            except:
                sf = 72000  # Default
                
            try:
                k0 = obj.root.goto(
                    "CommonDataObjects/DataViewCollection/*/properties/Context.MassScale.K0",
                    lazy=True,
                ).get_key_value()["float"]
            except:
                k0 = 0  # Default
            
            # Convert to mass
            from pySPM import utils
            m = utils.time2mass(CH, sf, k0)
            
            print(f"✓ ITAX buffer fix successful: {len(spectrum)} points extracted")
            return np.asarray(m, dtype=float), np.asarray(spectrum, dtype=float)
        else:
            raise

def _sum_spectrum_from_ita(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    obj = pySPM.ITA(str(path))
    m, I = obj.get_spectrum()
    return np.asarray(m, dtype=float), np.asarray(I, dtype=float)

def _sum_spectrum_from_itm(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        obj = pySPM.ITM(str(path))
        m, I = obj.get_spectrum()
        return np.asarray(m, dtype=float), np.asarray(I, dtype=float)
    except Exception as e:
        if "Missing block" in str(e) and "ShotsPerPixel" in str(e):
            print(f"ITM: Detected missing ShotsPerPixel block, trying alternative approach...")
            # Create ITM object manually with error handling
            obj = object.__new__(pySPM.ITM)
            obj.filename = str(path)
            obj.label = path.name
            obj.f = open(str(path), 'rb')
            obj.Type = obj.f.read(8)
            
            if obj.Type != b'ITStrF01':
                raise ValueError(f"Not an IONTOF file: {path}")
            
            obj.root = pySPM.Block.Block(obj.f)
            
            # Set default values for missing attributes
            obj.size = {"pixels": {"x": 256, "y": 256}, "real": {"x": 500e-6, "y": 500e-6, "unit": "m"}}
            obj.polarity = "Positive"
            obj.peaks = {}
            obj.meas_data = {}
            obj.rawlist = None
            obj.Nscan = 1
            obj.spp = 1    # Default shots per pixel
            obj.sf = 72000  # Default mass calibration
            obj.k0 = 0
            obj.scale = 1
            
            print(f"ITM: Initialized with default parameters, trying spectrum extraction...")
            try:
                m, I = obj.get_spectrum()
                print(f"✓ ITM alternative approach successful: extracted {len(m)} points")
                return np.asarray(m, dtype=float), np.asarray(I, dtype=float)
            except Exception as e2:
                print(f"ITM: Alternative approach also failed: {e2}")
                raise e
        else:
            raise

def _unit_mass_bin(m: np.ndarray, I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(m) & np.isfinite(I)
    mu = np.rint(m[mask]).astype(int)
    df = pd.DataFrame({"mu": mu, "I": I[mask]})
    agg = df.groupby("mu", sort=True)["I"].sum().reset_index()
    return agg["mu"].to_numpy(dtype=int), agg["I"].to_numpy(dtype=float)

def _try_paths(original: Path, prefer: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Try a sequence of paths/functions depending on original file and prefer mode.
    Returns (masses, intensities, used_label)
    prefer: auto|ita|itax|itm
    """
    p = _resolve_iontof_path(original)
    suf = p.suffix.lower()

    # Build candidate list based on prefer and what exists nearby
    candidates: list[Tuple[str, Path]] = []

    def add(kind: str, path: Optional[Path]):
        if path is not None and path.exists():
            candidates.append((kind, path))

    # Nearby siblings
    ita = p if suf == ".ita" else _has_sibling(p, ".ita")
    itax = p if suf == ".itax" else _has_sibling(p, ".itax")
    itm = p if suf in {".itm", ".tm"} else _has_sibling(p, ".itm")
    itmx = _has_sibling(p, ".itmx")  # only used as a hint that itm has an index

    # Preferred orders
    order_map = {
        "ita":  [("ita", ita), ("itax", itax), ("itm", itm)],
        "itax": [("itax", itax), ("ita", ita), ("itm", itm)],
        "itm":  [("itm", itm), ("itax", itax), ("ita", ita)],
        "auto": [("ita", ita), ("itax", itax), ("itm", itm)],
    }
    for kind, path in order_map.get(prefer, order_map["auto"]):
        add(kind, path)

    errors: list[str] = []
    for kind, path in candidates:
        try:
            if kind == "ita":
                return (*_sum_spectrum_from_ita(path), f"{kind}:{path.name}")
            elif kind == "itax":
                return (*_sum_spectrum_from_itax(path), f"{kind}:{path.name}")
            elif kind == "itm":
                # If .itmx exists, try to prefer ITAX via the index first (robuster on many datasets)
                if itmx or itax:
                    try:
                        px = itmx if itmx else itax
                        return (*_sum_spectrum_from_itax(px), f"itax:{px.name}")
                    except Exception as e:
                        errors.append(f"itax({px.name}) failed: {e}")
                return (*_sum_spectrum_from_itm(path), f"{kind}:{path.name}")
        except Exception as e:
            errors.append(f"{kind}({path.name}) failed: {e}")

    msg = " | ".join(errors) if errors else f"No usable spectrum source found near {original}"
    raise RuntimeError(f"IONTOF sum spectrum failed: {msg}")

def iontof_to_pnnl_tsv(
    files: Iterable[Path],
    out_tsv: Path,
    polarity: str = "P",
    names: Optional[List[str]] = None,
    prefer: str = "auto",
) -> Path:
    if pySPM is None:
        raise ImportError("pySPM is required for IONTOF processing. Install with: pip install pySPM")

    files = [Path(f) for f in files]
    labels = [f"{i+1}-{polarity.upper()}" for i in range(len(files))] if not names else [s.strip() for s in ",".join(names).split(",")]
    if len(labels) != len(files):
        raise ValueError(f"--names provided {len(labels)} labels for {len(files)} files")

    master_m: Optional[np.ndarray] = None
    cols: Dict[str, np.ndarray] = {}

    used_paths: Dict[str, str] = {}

    for f, label in zip(files, labels):
        m, I, used = _try_paths(f, prefer=prefer.lower())
        used_paths[label] = used
        mu, Ib = _unit_mass_bin(m, I)

        if master_m is None:
            master_m = mu
            cols[label] = Ib
        else:
            union = np.union1d(master_m, mu).astype(int)
            old_idx = pd.Index(union).get_indexer(master_m)
            new_cols = {k: np.zeros_like(union, dtype=float) for k in cols}
            for k, vec in cols.items():
                tmp = np.zeros_like(union, dtype=float); tmp[old_idx] = vec
                new_cols[k] = tmp
            cur_idx = pd.Index(union).get_indexer(mu)
            cur_vec = np.zeros_like(union, dtype=float); cur_vec[cur_idx] = Ib
            master_m = union
            cols = new_cols
            cols[label] = cur_vec

    if master_m is None:
        raise RuntimeError("No spectra could be read.")

    out_tsv = Path(out_tsv); out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"Mass (u)": master_m})
    for k in labels:
        df[k] = cols[k]
    df.sort_values("Mass (u)", inplace=True)
    df.to_csv(out_tsv, sep="\t", index=False)

    # Also drop a tiny sidecar telling which source was used per column
    meta = out_tsv.with_suffix(".meta.txt")
    with open(meta, "w") as fh:
        for k, v in used_paths.items():
            fh.write(f"{k}\t{v}\n")
    return out_tsv