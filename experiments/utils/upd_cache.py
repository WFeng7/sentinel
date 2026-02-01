import json
import re
import math
import hashlib
from collections import defaultdict
from difflib import SequenceMatcher

import requests

RIDOT_QUERY_URL = "https://vueworks.dot.ri.gov/arcgis/rest/services/VW_ITSAssets105/MapServer/2/query"

INPUT_CAMERAS = [
  {
    "id": "cam15",
    "label": "95-27 I-95 S @ Garden",
    "stream": "https://cdn3.wowza.com/1/bUpoYVNwZlBIaTAw/dzVjWm9O/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4188
  },
  {
    "id": "cam37",
    "label": "Atwells Ave",
    "stream": "https://cdn3.wowza.com/1/aUdHZFJSUy91dkp4/UzFvOTFW/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4188
  },
  {
    "id": "cam38",
    "label": "Dean Street",
    "stream": "https://cdn3.wowza.com/1/Y2ZmK1ErdG1zUXZF/TWJFdzNI/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4128
  },
  {
    "id": "cam43",
    "label": "DMS and Camera Rt 146 SB",
    "stream": "https://cdn3.wowza.com/1/bHQzelM2aHlNZlRm/L2N5SHg2/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam34",
    "label": "Flyover",
    "stream": "https://cdn3.wowza.com/1/NHcwZTF1K2s3NGpu/Zzc4eTht/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4188
  },
  {
    "id": "cam33",
    "label": "Hartford Ave",
    "stream": "https://cdn3.wowza.com/1/WE5TRUNFT2lDUWg5/aFE3eU1K/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4188
  },
  {
    "id": "cam24",
    "label": "I-195 E @ India St.",
    "stream": "https://cdn3.wowza.com/1/UEVONS9LdlVTcXVs/ZkNVS3Rq/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam26",
    "label": "I-195 E @ Point St.",
    "stream": "https://cdn3.wowza.com/1/Qis0Y1U4bUJxeEtX/TEROUGFP/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam21",
    "label": "I-195 E @ Rt 114",
    "stream": "https://cdn3.wowza.com/1/Rm8zcEM0NFRwMk1J/MFVNUlhk/hls/live/playlist.m3u8",
    "lat": 41.8084,
    "lng": -71.4482
  },
  {
    "id": "cam23",
    "label": "I-195 W @ Gano St",
    "stream": "https://cdn3.wowza.com/1/K2dGTStHZjhqZkhD/aTNZenFh/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam20",
    "label": "I-195 W @ Mass State Line",
    "stream": "https://cdn3.wowza.com/1/S25rZ2lObHFOL1Zo/R21qVzhK/hls/live/playlist.m3u8",
    "lat": 41.7687,
    "lng": -71.4482
  },
  {
    "id": "cam22",
    "label": "I-195 W @ Washington Bridge",
    "stream": "https://cdn3.wowza.com/1/VkxzemhLak1HL2tO/Z01ENHl0/hls/live/playlist.m3u8",
    "lat": 41.8234,
    "lng": -71.4189
  },
  {
    "id": "cam25",
    "label": "I-195 WB Split @ I-95",
    "stream": "https://cdn3.wowza.com/1/T05XOENNUVZBQ0cr/STJqWUVl/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam7",
    "label": "I-95 @ Broadway (Prov)",
    "stream": "https://cdn3.wowza.com/1/K1BOYVhLYlZQVzEz/U2taVnRF/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam11",
    "label": "I-95 N @ Branch Ave",
    "stream": "https://cdn3.wowza.com/1/K1g4UTlTQzF5ZGdV/TEhZNCs0/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam17",
    "label": "I-95 N @ Broadway",
    "stream": "https://cdn3.wowza.com/1/SW4yU0tJN2hmVUNZ/THh0V2h6/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam9",
    "label": "I-95 N @ Kinsley Ave",
    "stream": "https://cdn3.wowza.com/1/UVZwZFN0R2U1bGNn/OWhQMi9K/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam14",
    "label": "I-95 N @ Lonsdale",
    "stream": "https://cdn3.wowza.com/1/M2I4Mm5OdGxvQ3g4/SGZjcUxN/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam5",
    "label": "I-95 N @ Public St",
    "stream": "https://cdn3.wowza.com/1/L3E4Z0NpRWZkWEhZ/SDQ5dVdX/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam12",
    "label": "I-95 N @ Smithfield Ave",
    "stream": "https://cdn3.wowza.com/1/ZTdLZmtEVnB1aVEz/M3lZck51/hls/live/playlist.m3u8",
    "lat": 41.8781,
    "lng": -71.4482
  },
  {
    "id": "cam13",
    "label": "I-95 N @ Smithfield Ave",
    "stream": "https://cdn3.wowza.com/1/bTRwbVZrNUlKOUx3/b3pHWWlE/hls/live/playlist.m3u8",
    "lat": 41.8781,
    "lng": -71.4482
  },
  {
    "id": "cam4",
    "label": "I-95 N @ Thurbers Ave",
    "stream": "https://cdn3.wowza.com/1/ZDdEQkhqNDFkTk1u/eHFMUk4y/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4154
  },
  {
    "id": "cam16",
    "label": "I-95 N @ Vernon St",
    "stream": "https://cdn3.wowza.com/1/TzB2dzNuaGgybVFs/MndKekth/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam8",
    "label": "I-95 S @ 6/10 Interchange",
    "stream": "https://cdn3.wowza.com/1/cnFkMExDMWdpbUdv/UC9mOXlD/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4188
  },
  {
    "id": "cam6",
    "label": "I-95 S @ Broad St",
    "stream": "https://cdn3.wowza.com/1/c2FPYjV5NENIcC9C/Q29XaGtz/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam19",
    "label": "I-95 S @ Central Ave",
    "stream": "https://cdn3.wowza.com/1/Tk9TUXZtelhITzdz/YlhCTjNa/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam3",
    "label": "I-95 S @ Detroit Ave",
    "stream": "https://cdn3.wowza.com/1/SmhURWlnU0pveTM2/Mm5Dak1W/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4188
  },
  {
    "id": "cam18",
    "label": "I-95 S @ East St",
    "stream": "https://cdn3.wowza.com/1/eGhaTjRuL3l3bGg5/M1FCVDRq/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam2",
    "label": "I-95 S @ Elmwood Ave",
    "stream": "https://cdn3.wowza.com/1/SDlmZ3dZaUkyakVl/VkJrR2Zw/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4174
  },
  {
    "id": "cam10",
    "label": "I-95 S @ Orms St",
    "stream": "https://cdn3.wowza.com/1/cHVVQ21DWGJ2Qnpy/bDhTTVlD/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4188
  },
  {
    "id": "cam1",
    "label": "I-95 S @ Rt 10",
    "stream": "https://cdn3.wowza.com/1/eUp6WUZ2Q0NiTnNh/K2VvaWJo/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4188
  },
  {
    "id": "cam28",
    "label": "Memorial @ Francis St",
    "stream": "https://cdn3.wowza.com/1/LzhLdUlsTHQ5aG9k/YVZybnlS/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam27",
    "label": "Memorial @ Steeple",
    "stream": "https://cdn3.wowza.com/1/U24raXhhbDN3UWJH/elBINE5K/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4128
  },
  {
    "id": "cam29",
    "label": "Promenade St @ Dean St",
    "stream": "https://cdn3.wowza.com/1/TGF3cncxekFxazFM/Ymg2T1Ev/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam39",
    "label": "Rt 10 N @ Elmwood",
    "stream": "https://cdn3.wowza.com/1/R3FpanNleVVGTS90/RG9OQ1M3/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4348
  },
  {
    "id": "cam41",
    "label": "Rt 10 N @ Kenwood Ave",
    "stream": "https://cdn3.wowza.com/1/d01Za3draDF4NnJt/Z0plK1o2/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4188
  },
  {
    "id": "cam40",
    "label": "Rt 10 S @ Reservoir",
    "stream": "https://cdn3.wowza.com/1/c2JzN2VTUEFwMjBh/YmM5UWMx/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam42",
    "label": "Rt 10 S @ Union Ave",
    "stream": "https://cdn3.wowza.com/1/RWNoZFBOSVVEdzB3/WDZyOEx1/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam44",
    "label": "Rt 146 NB @ Rt 116",
    "stream": "https://cdn3.wowza.com/1/N0FHOUcvRHNpQUMy/QUZyYVB4/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4482
  },
  {
    "id": "cam48",
    "label": "Rt 146 North at Admiral Street",
    "stream": "https://cdn3.wowza.com/1/SVlYZ2c0cFdia1I4/RHk5YVZs/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4482
  },
  {
    "id": "cam47",
    "label": "Rt 146 North at Charles Street",
    "stream": "https://cdn3.wowza.com/1/eHZKc2lEaThhNEtJ/bWU4RlpI/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam45",
    "label": "Rt 146 North at Sherman Ave",
    "stream": "https://cdn3.wowza.com/1/SkRQeFhmUk9sTDJG/dkovKzdK/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4484
  },
  {
    "id": "cam49",
    "label": "Rt 146 North State Police",
    "stream": "https://cdn3.wowza.com/1/anByd1AxSVVzSVUw/aDZQZXdS/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam46",
    "label": "Rt 146 S @ Branch Ave",
    "stream": "https://cdn3.wowza.com/1/LzhZSERKYXgySVVU/SGY3d1RH/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4482
  },
  {
    "id": "cam30",
    "label": "Rt 6 W @ Atwood",
    "stream": "https://cdn3.wowza.com/1/RERYRTlUL2VHdW11/VjZ0cEN4/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam31",
    "label": "Rt 6 W @ Glenbridge",
    "stream": "https://cdn3.wowza.com/1/ZXFXUkJFZ2NSV3BO/QkVwdlFt/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam32",
    "label": "Rt 6 W DMS and Camera",
    "stream": "https://cdn3.wowza.com/1/QStJM3lUL3hrRmsw/RDhlL0py/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4188
  },
  {
    "id": "cam36",
    "label": "Tobey Street",
    "stream": "https://cdn3.wowza.com/1/eVFWVWpSY3VyWVRa/QVJ4OGV0/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4134
  },
  {
    "id": "cam35",
    "label": "Westminster Street",
    "stream": "https://cdn3.wowza.com/1/NlpMMjR5bXFWNk40/NFF6ME16/hls/live/playlist.m3u8",
    "lat": 41.8231,
    "lng": -71.4128
  }
]

# --- normalization / scoring ---

STOP = {
    "at", "and", "camera", "dms", "the", "of", "on", "in", "to",
    "north", "south", "east", "west", "nb", "sb", "eb", "wb",
    "n", "s", "e", "w",
    "st", "street", "ave", "avenue", "rd", "road", "blvd", "boulevard",
    "route", "rt", "i", "interchange"
}

DIR_MAP = {
    "nb": "northbound", "sb": "southbound", "eb": "eastbound", "wb": "westbound",
    "n": "northbound", "s": "southbound", "e": "eastbound", "w": "westbound",
}

def norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("@", " at ")
    s = s.replace("/", " ")
    s = re.sub(r"\brt\b", "route", s)
    s = re.sub(r"\bi\s*-\s*", "i-", s)
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str):
    t = []
    for w in norm_text(s).split():
        if w in DIR_MAP:
            w = DIR_MAP[w]
        if w in STOP:
            continue
        t.append(w)
    return t

def token_jaccard(a, b) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def score_match(cam_label: str, ridot_desc: str, ridot_dir: str = "") -> float:
    # combine token overlap + sequence similarity
    ca = tokens(cam_label)
    rb = tokens(ridot_desc)

    j = token_jaccard(ca, rb)
    r = seq_ratio(norm_text(cam_label), norm_text(ridot_desc))

    # slight bonus if direction seems consistent
    bonus = 0.0
    cl = norm_text(cam_label)
    d = (ridot_dir or "").lower()
    if d:
        if ("north" in cl or "nb" in cl or "n " in cl) and ("north" in d or "nb" in d):
            bonus = 0.05
        if ("south" in cl or "sb" in cl or "s " in cl) and ("south" in d or "sb" in d):
            bonus = 0.05
        if ("east" in cl or "eb" in cl or "e " in cl) and ("east" in d or "eb" in d):
            bonus = 0.05
        if ("west" in cl or "wb" in cl or "w " in cl) and ("west" in d or "wb" in d):
            bonus = 0.05

    # weighted blend; jaccard drives “same place words”, ratio catches punctuation/format differences
    return 0.65 * j + 0.35 * r + bonus

# --- ridot fetch ---

def fetch_ridot():
    params = {
        "where": "1=1",
        "outFields": "Description,Latitude,Longitude,Direction,EquipmentID,AimetisID,CCVEWebURL",
        "returnGeometry": "false",
        "f": "json",
    }
    r = requests.get(RIDOT_QUERY_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    rows = []
    for f in data.get("features", []):
        a = f.get("attributes") or {}
        lat = a.get("Latitude")
        lng = a.get("Longitude")
        desc = (a.get("Description") or "").strip()
        if not desc or lat is None or lng is None:
            continue
        rows.append({
            "desc": desc,
            "dir": a.get("Direction") or "",
            "lat": float(lat),
            "lng": float(lng),
            "equip": a.get("EquipmentID"),
            "aim": a.get("AimetisID") or "",
            "url": a.get("CCVEWebURL") or "",
        })
    return rows

# --- jitter only if needed (deterministic, tiny) ---

def deterministic_jitter(cam_id: str, eps_deg: float = 1e-5):
    """
    eps_deg=1e-5 ~ 1.1m latitude. This is only used if exact dupes remain.
    """
    h = hashlib.sha256(cam_id.encode()).hexdigest()
    # map to [-1,1]
    a = (int(h[:8], 16) / 0xFFFFFFFF) * 2 - 1
    b = (int(h[8:16], 16) / 0xFFFFFFFF) * 2 - 1
    return a * eps_deg, b * eps_deg

def iter_cams(x):
    # Accept either: [ {..}, {..} ] or [ [ {..}, {..} ] ]
    if not x:
        return
    if isinstance(x[0], dict):
        yield from x
    elif isinstance(x[0], list):
        for inner in x:
            yield from inner
    else:
        raise TypeError(f"Unexpected INPUT_CAMERAS shape: first element is {type(x[0])}")

def main():
    ridot = fetch_ridot()
    if not ridot:
        raise RuntimeError("No RIDOT camera rows returned (network? service down?)")

    updated = []
    diagnostics = []

    for cam in iter_cams(INPUT_CAMERAS):
        label = cam.get("label", "")
        best = None
        best_s = -1.0

        for r in ridot:
            s = score_match(label, r["desc"], r["dir"])
            if s > best_s:
                best_s = s
                best = r

        cam2 = dict(cam)

        # threshold: tune if needed
        # - if you set too high, you’ll get unmatched; too low risks bad matches
        THRESH = 0.55
        matched = best is not None and best_s >= THRESH

        if matched:
            cam2["lat"] = round(best["lat"], 6)
            cam2["lng"] = round(best["lng"], 6)
            diagnostics.append((cam.get("id"), label, best["desc"], best_s))
        else:
            diagnostics.append((cam.get("id"), label, None, best_s))

        updated.append(cam2)

    # dupe check
    coord_map = defaultdict(list)
    for c in updated:
        coord_map[(c["lat"], c["lng"])].append(c["id"])
    dupes = {k: v for k, v in coord_map.items() if len(v) > 1}

    # If dupes remain, jitter all but first in each group
    if dupes:
        id_to_idx = {c["id"]: i for i, c in enumerate(updated)}
        for (lat, lng), ids in dupes.items():
            ids_sorted = sorted(ids)
            for cam_id in ids_sorted[1:]:
                i = id_to_idx[cam_id]
                dlat, dlng = deterministic_jitter(cam_id, eps_deg=1e-5)
                updated[i]["lat"] = round(updated[i]["lat"] + dlat, 6)
                updated[i]["lng"] = round(updated[i]["lng"] + dlng, 6)

        # recompute dupes after jitter
        coord_map2 = defaultdict(list)
        for c in updated:
            coord_map2[(c["lat"], c["lng"])].append(c["id"])
        dupes2 = {k: v for k, v in coord_map2.items() if len(v) > 1}
    else:
        dupes2 = {}

    # ---- OUTPUT JSON ----
    print("\n===== UPDATED CAMERA CACHE =====\n")
    print(json.dumps(updated, indent=2))

    # ---- SUMMARY ----
    matched = [d for d in diagnostics if d[2] is not None]
    unmatched = [d for d in diagnostics if d[2] is None]

    print("\n===== MATCH DIAGNOSTICS =====")
    print(f"Matched: {len(matched)} / {len(updated)}")
    print(f"Unmatched: {len(unmatched)} / {len(updated)}")

    # show worst 15 unmatched (highest scores but below threshold) to tune THRESH if needed
    unmatched_sorted = sorted(unmatched, key=lambda x: x[3], reverse=True)[:15]
    if unmatched_sorted:
        print("\nTop unmatched candidates (score, id, label):")
        for cam_id, label, _, s in unmatched_sorted:
            print(f"  {s:.3f}  {cam_id}  {label}")

    # show 15 weakest matches to sanity check for false positives
    matched_sorted = sorted(matched, key=lambda x: x[3])[:15]
    if matched_sorted:
        print("\nWeakest matches (score, id, your_label -> ridot_desc):")
        for cam_id, label, desc, s in matched_sorted:
            print(f"  {s:.3f}  {cam_id}  {label}  ->  {desc}")

    print("\n===== DUPES =====")
    print(f"Duplicate groups before jitter: {len(dupes)}")
    if dupes:
        for (lat, lng), ids in dupes.items():
            print(f"  ({lat}, {lng}) -> {ids}")

    print(f"Duplicate groups after jitter: {len(dupes2)}")
    if dupes2:
        for (lat, lng), ids in dupes2.items():
            print(f"  ({lat}, {lng}) -> {ids}")

if __name__ == "__main__":
    main()
