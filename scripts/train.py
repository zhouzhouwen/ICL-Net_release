import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATASET_SHA256 = "7bcd8cf5530e8752455b374e6d723eb947457323d4f203ecea8cdf89ec46a070"
FEATURE_SCHEMA_SHA256 = "eb492045af18887e21abb22441519bab6682341bfded71a0dc25903102bce7c5"
SPLIT_SEED = 20260929
SEEDS = (42, 43, 44)
WINDOWS_MIN = (5, 8, 10, 11, 12, 13, 14, 15, 20)
TUNING_WINDOWS_MIN = (10, 12, 15)
OUTPUT_TIMES = np.r_[np.arange(0, 1981, 60), 2000].astype(np.int64)
BREAK_COLUMNS = (14, 15, 16)
INPUT_COLUMNS = tuple(i for i in range(1, 38) if i not in BREAK_COLUMNS)
EXTRA_IDS = ("grape-sgtr-i_025075_b", "grape-sgtr-o_045065_e", "grape-sgtr-t_035075_e")
GROUPS = (
    ("Neutronics", (0, 1)),
    ("Coolant temperature", (3, 4, 5, 6, 7, 8)),
    ("Primary pressure and level", (9, 10, 11, 12)),
    ("Charging and letdown", (13, 14, 15)),
    ("Safety injection", (2, 16, 17)),
    ("SG pressure and level", (18, 19, 20, 21)),
    ("Valve status", (22, 23, 24, 25)),
    ("Feedwater", (26, 27, 28, 30, 31, 32)),
    ("Steam flows", (29, 33)),
)
CONDITIONS = ("clean", "drop1", "drop3", "drop7", *(f"group_{i}" for i in range(9)), "block7", "outage7")
RANDOM_CONDITIONS = frozenset(("drop1", "drop3", "drop7", "block7", "outage7"))
CHANNELS = (
    ("neutron_flux_pct", "Power-range neutron flux", "%"),
    ("reactor_trip", "Reactor trip signal", "binary"),
    ("si_signal", "Safety-injection actuation signal", "binary"),
    ("t_cold_a_degc", "Loop A cold-leg temperature", "°C"),
    ("t_hot_a_degc", "Loop A hot-leg temperature", "°C"),
    ("t_avg_a_degc", "Loop A average coolant temperature", "°C"),
    ("t_avg_b_degc", "Loop B average coolant temperature", "°C"),
    ("t_hot_b_degc", "Loop B hot-leg temperature", "°C"),
    ("t_cold_b_degc", "Loop B cold-leg temperature", "°C"),
    ("p_rcs_b_pa", "Loop B primary coolant pressure", "Pa"),
    ("prv_open", "Pressurizer relief valve opening", "fraction"),
    ("p_prz_pa", "Pressurizer pressure", "Pa"),
    ("l_prz_pct", "Pressurizer level", "% span"),
    ("charging_a_kgs", "Loop A charging flow", "kg/s"),
    ("charging_b_kgs", "Loop B charging flow", "kg/s"),
    ("letdown_kgs", "Letdown flow", "kg/s"),
    ("si_a_kgs", "Loop A safety-injection flow", "kg/s"),
    ("si_b_kgs", "Loop B safety-injection flow", "kg/s"),
    ("p_sg_a_pa", "SG A pressure", "Pa"),
    ("p_sg_b_pa", "SG B pressure", "Pa"),
    ("l_sg_a_pct", "SG A narrow-range level", "% span"),
    ("l_sg_b_pct", "SG B narrow-range level", "% span"),
    ("msiv_a_s", "SG A MSIV closure time (-1 before closure)", "s"),
    ("msiv_b_s", "SG B MSIV closure time (-1 before closure)", "s"),
    ("msrv_a", "SG A MSRV opening", "fraction"),
    ("msrv_b", "SG B MSRV opening", "fraction"),
    ("mfw_a_kgs", "Loop A main feedwater flow", "kg/s"),
    ("mdafw_a_kgs", "Loop A motor-driven AFW flow", "kg/s"),
    ("tdafw_a_kgs", "Loop A turbine-driven AFW flow", "kg/s"),
    ("ms_a_kgs", "Loop A main steam flow", "kg/s"),
    ("mfw_b_kgs", "Loop B main feedwater flow", "kg/s"),
    ("mdafw_b_kgs", "Loop B motor-driven AFW flow", "kg/s"),
    ("tdafw_b_kgs", "Loop B turbine-driven AFW flow", "kg/s"),
    ("ms_b_kgs", "Loop B main steam flow", "kg/s"),
)
PHASES = 60
BATCH_SIZE = 32
PREDICT_BATCH_SIZE = 256
MAX_EPOCHS = 240
PATIENCE = 30
MIN_EPOCHS = 60
GRADIENT_CLIP = 1.0
FULL_CURVE_WEIGHT = 0.1
COVERAGE = 0.9
SEARCH = dict(
    trials=48, startup_trials=12, sampler_seed=42,
    learning_rate=(5e-5, 5e-3), weight_decay=(0.0, 1e-6, 1e-5, 1e-4, 1e-3),
    width=(64, 96, 128, 192, 256), dropout=(0.0, 0.1, 0.2, 0.3, 0.4), depth=(1, 2, 3),
    decoder=("recurrent_direct", "recurrent_spline", "recurrent_segmented"),
    auxiliary_mode=("state_anchor", "state_anchor_shape"), auxiliary_weight=(0.02, 0.4),
)
PAPER_HYPERPARAMETERS = {
    0: dict(width=256, depth=1, dropout=0.1, decoder="recurrent_spline", learning_rate=0.00015249626616793766,
            weight_decay=1e-06, auxiliary_mode="state_anchor_shape", auxiliary_weight=0.29454809777609375, epochs=189),
    1: dict(width=128, depth=1, dropout=0.1, decoder="recurrent_segmented", learning_rate=0.00012279379430121036,
            weight_decay=0.0, auxiliary_mode="state_anchor_shape", auxiliary_weight=0.36089062363734414, epochs=111),
    2: dict(width=256, depth=2, dropout=0.1, decoder="recurrent_segmented", learning_rate=0.0008122172859640715,
            weight_decay=1e-05, auxiliary_mode="state_anchor_shape", auxiliary_weight=0.3471708823345346, epochs=204),
    3: dict(width=256, depth=1, dropout=0.0, decoder="recurrent_segmented", learning_rate=7.388612494692168e-05,
            weight_decay=1e-06, auxiliary_mode="state_anchor", auxiliary_weight=0.04631490328176148, epochs=150),
    4: dict(width=192, depth=1, dropout=0.1, decoder="recurrent_segmented", learning_rate=0.00012321357654957468,
            weight_decay=1e-06, auxiliary_mode="state_anchor", auxiliary_weight=0.3353038736939957, epochs=178),
}


def _hash(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _rng(*parts):
    return np.random.default_rng(int(_hash(*parts)[:16], 16))


def load_database(data_dir):
    root = Path(data_dir).resolve()
    label_path = root / "spds.plt"
    if not label_path.is_file():
        raise SystemExit(f"Dataset not found in {root}. The data are confidential; see data/README.md.")
    labels = ["time_s"] + [line.split("*", 1)[1].strip()
                           for line in label_path.read_text(encoding="utf-8").splitlines() if "*" in line]
    if len(labels) != 38 or any(not labels[i].startswith("break flow") for i in BREAK_COLUMNS):
        raise ValueError("spds.plt does not match the 38-column schema")
    records, inputs, targets = [], [], []
    for path in sorted(root.glob("*.spds.prn")):
        case_id = path.name[:-len(".spds.prn")]
        match = re.fullmatch(r"grape-sgtr-([iot])_(\d{2})(\d{2})_([be])", case_id)
        extra = re.fullmatch(r"grape-sgtr-([iot])_(\d{3})(\d{3})_([be])", case_id)
        if match:
            location, isolation, relief, core = match.groups()
            isolation, relief, is_extra = float(isolation), float(relief), False
        elif case_id in EXTRA_IDS and extra:
            location, isolation, relief, core = extra.groups()
            isolation, relief, is_extra = float(isolation) / 10, float(relief) / 10, True
        else:
            raise ValueError(f"Unrecognised case file: {path.name}")
        a = np.loadtxt(path, skiprows=2, dtype=np.float64)
        if a.shape != (2001, 38) or not np.isfinite(a).all() or not np.array_equal(a[:, 0], np.arange(2001)):
            raise ValueError(f"Invalid raw file: {path.name}")
        positive = np.maximum(a[:, BREAK_COLUMNS].sum(axis=1), 0.0)
        if not np.allclose(positive, np.maximum(a[:, BREAK_COLUMNS], 0).sum(axis=1), atol=1e-9, rtol=0):
            raise ValueError(f"Break-flow conventions diverge: {path.name}")
        mass = np.r_[0.0, np.cumsum(positive[1:], dtype=np.float64)]
        events = []
        for column in (2, 26, 29):
            changed = np.flatnonzero(a[:, column] != a[0, column])
            events.append(float(changed[0]) if len(changed) else float("nan"))
        if events[1] - events[0] != isolation * 60 or events[2] - events[0] != relief * 60:
            raise ValueError(f"Operation timing mismatch: {path.name}")
        inputs.append(a[:, INPUT_COLUMNS].astype(np.float32))
        targets.append(mass[OUTPUT_TIMES])
        records.append(dict(case_id=case_id, location=location, core=core, isolation_min=isolation,
                            relief_min=relief, is_extra=is_extra,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    factorial = {(r["location"], r["isolation_min"], r["relief_min"], r["core"]) for r in records if not r["is_extra"]}
    if factorial != set(itertools.product("iot", range(1, 6), range(6, 11), "be")) or \
            sum(not r["is_extra"] for r in records) != 150:
        raise ValueError("The dataset must contain the complete 150-case factorial design")
    data_hash = _hash(hashlib.sha256(label_path.read_bytes()).hexdigest(),
                      [[r["case_id"], r["sha256"]] for r in records], "positive_sum_right_1s_initial_zero")
    return dict(case_ids=[r["case_id"] for r in records], meta=records, data_hash=data_hash,
                inputs=np.ascontiguousarray(np.stack(inputs)), targets=np.stack(targets))


def make_folds(db, seed=SPLIT_SEED):
    meta = db["meta"]
    fold_map = {r["case_id"]: -1 for r in meta}
    rng = np.random.default_rng(seed)
    for location, core in itertools.product("iot", "be"):
        group = [r for r in meta if r["location"] == location and r["core"] == core and not r["is_extra"]]
        pair_order = rng.permutation(np.arange(1, 6))
        singles = np.array(sorted(r["case_id"] for r in group if r["relief_min"] not in (6, 10)))
        singles = singles[rng.permutation(len(singles))].reshape(5, 3)
        for fold, isolation in enumerate(pair_order):
            pair = [r["case_id"] for r in group if r["isolation_min"] == isolation and r["relief_min"] in (6, 10)]
            for name in pair + list(singles[fold]):
                fold_map[name] = fold
    return fold_map


def fold_roles(db, fold_map, fold):
    roles = dict(test=(fold,), cal=((fold + 1) % 5,), val=((fold + 2) % 5,),
                 train=((fold + 3) % 5, (fold + 4) % 5), fit=((fold + 2) % 5, (fold + 3) % 5, (fold + 4) % 5))
    return {role: [n for n in db["case_ids"] if fold_map[n] in folds] for role, folds in roles.items()}


class Scaler:
    def __init__(self, input_min, input_max, fill_mean, target_scale):
        self.input_min = np.asarray(input_min, dtype=np.float64)
        self.input_max = np.asarray(input_max, dtype=np.float64)
        self.fill_mean = np.asarray(fill_mean, dtype=np.float64)
        self.target_scale = float(target_scale)

    @property
    def scale(self):
        span = self.input_max - self.input_min
        return np.where(span > 0, span, 1.0)

    def transform(self, values):
        return ((values - self.input_min) / self.scale).astype(np.float32)

    def to_dict(self):
        return dict(input_min=self.input_min.tolist(), input_max=self.input_max.tolist(),
                    fill_mean=self.fill_mean.tolist(), target_scale=self.target_scale)


def fit_scaler(db, case_ids):
    index = {n: i for i, n in enumerate(db["case_ids"])}
    values = np.ascontiguousarray(db["inputs"][[index[c] for c in case_ids]])
    targets = db["targets"][[index[c] for c in case_ids]]
    return Scaler(values.min(axis=(0, 1)).astype(np.float64), values.max(axis=(0, 1)).astype(np.float64),
                  values.mean(axis=(0, 1), dtype=np.float64), float(targets[:, -1].max()))


def fault_spec(case_id, condition, repeat=0, seed=SPLIT_SEED):
    if condition == "clean":
        return [], -1, -1
    if condition.startswith("group_"):
        return list(GROUPS[int(condition.split("_")[1])][1]), 0, 2001
    n = int(condition[4:]) if condition.startswith("drop") else 7
    channels = sorted(_rng(seed, "fault", case_id, condition, repeat).choice(34, n, replace=False).tolist())
    start, end = (300, 420) if condition == "block7" else ((600, 2001) if condition == "outage7" else (0, 2001))
    return channels, start, end


def sample_times(window_min, phase=0):
    return np.unique(np.r_[0, np.arange(phase, int(window_min) * 60 + 1, 60)]).astype(np.int64)


def observation_tensor(values, times, window_min, scaler, available):
    previous = np.maximum.accumulate(np.where(available, np.arange(len(times))[:, None], -1), axis=0)
    filled = np.where(previous >= 0, values[np.maximum(previous, 0), np.arange(34)], scaler.fill_mean)
    result = np.zeros((21, 71), dtype=np.float32)
    n = len(times)
    result[:n, :34] = scaler.transform(filled)
    result[:n, 34:68] = available
    result[:n, 68] = 1
    result[:n, 69] = times / 2000.0
    result[:n, 70] = float(window_min) * 60 / 2000.0
    return result


def build_samples(db, case_ids, scaler, windows=WINDOWS_MIN, phases=(0,), training=False,
                  condition="clean", seed=SPLIT_SEED):
    index = {n: i for i, n in enumerate(db["case_ids"])}
    conditions = (condition,) if isinstance(condition, str) else tuple(condition)
    base = [(index[name], name, p, w) for name in case_ids for p in phases for w in windows]
    augmented = set()
    if training:
        order = sorted(range(len(base)), key=lambda j: _hash(seed, "augment", *base[j][1:]))
        augmented = set(order[:len(base) // 2])
    specs = []
    for j, (i, name, phase, w) in enumerate(base):
        if training:
            cond, rep = "clean", 0
            if j in augmented:
                rng = _rng(seed, "family", name, phase, w)
                family = int(rng.integers(4))
                if family == 0:
                    cond = f"drop{rng.choice((1, 3, 7))}"
                elif family == 1:
                    cond = f"group_{rng.integers(9)}"
                else:
                    cond = ("block7", "outage7")[family - 2]
                rep = phase
            pairs = [(cond, rep)]
        else:
            pairs = [(c, r) for c in conditions for r in range(10 if c in RANDOM_CONDITIONS else 1)]
        specs.extend((i, name, phase, w, c, r) for c, r in pairs)
    X = np.empty((len(specs), 21, 71), dtype=np.float32)
    y = np.empty((len(specs), 35), dtype=np.float32)
    meta, cache = [], {}
    for j, (i, name, phase, w, cond, rep) in enumerate(specs):
        times = sample_times(w, phase)
        if (name, cond, rep) not in cache:
            cache[(name, cond, rep)] = fault_spec(name, cond, rep, seed)
        channels, start, end = cache[(name, cond, rep)]
        available = np.ones((len(times), 34), dtype=bool)
        if channels:
            available[np.ix_((times >= start) & (times < end), channels)] = False
        X[j] = observation_tensor(db["inputs"][i][times], times, w, scaler, available)
        y[j] = db["targets"][i] / scaler.target_scale
        meta.append((name, w, phase, cond, rep))
    return X, y, meta


EVENT_INDICES = (1, 2, 10, 22, 23, 24, 25)
N_INTERVALS = 34
N_SPLINE_KNOTS = 12
PHYSICAL_SPECS = (
    (((9, 1.), (18, -1.)), 0., 1e7),
    (((9, 1.), (19, -1.)), 0., 1e7),
    (((11, 1.), (18, -1.)), 0., 1e7),
    (((18, 1.), (19, -1.)), 0., 1e7),
    (((4, 1.), (3, -1.)), 0., 300.),
    (((7, 1.), (8, -1.)), 0., 300.),
    (((5, 1.), (6, -1.)), 0., 300.),
    (((7, 1.),), -276., 300.),
    (((20, 1.), (21, -1.)), 0., 100.),
    (((12, 1.),), 0., 100.),
    (((13, 1.), (14, 1.), (16, 1.), (17, 1.), (15, -1.)), 0., 100.),
    (((26, 1.), (27, 1.), (28, 1.), (29, -1.)), 0., 1500.),
    (((30, 1.), (31, 1.), (32, 1.), (33, -1.)), 0., 1500.),
    (((16, 1.), (17, 1.)), 0., 100.),
)


def _physical_constants(device, dtype):
    coefficients = torch.zeros((34, len(PHYSICAL_SPECS)), device=device, dtype=dtype)
    requirements = torch.zeros_like(coefficients)
    for j, (terms, _, _) in enumerate(PHYSICAL_SPECS):
        for i, c in terms:
            coefficients[i, j] = c
            requirements[i, j] = 1.
    offsets = torch.tensor([s[1] for s in PHYSICAL_SPECS], device=device, dtype=dtype)
    divisors = torch.tensor([s[2] for s in PHYSICAL_SPECS], device=device, dtype=dtype)
    return coefficients, requirements, offsets, divisors


def _history(values, available, times, valid, window):
    length, channels = values.shape[1], values.shape[2]
    positions = torch.arange(length, device=values.device).view(1, length, 1)
    last_index = torch.where(available, positions, -torch.ones_like(positions)).cummax(dim=1).values
    prior_index = torch.cat((torch.full_like(last_index[:, :1], -1), last_index[:, :-1]), dim=1)
    time_channels = times.expand(-1, -1, channels)
    previous_value = values.gather(1, prior_index.clamp_min(0))
    previous_time = time_channels.gather(1, prior_index.clamp_min(0))
    difference = (values - previous_value) * (60. / 2000.) / (time_channels - previous_time).clamp_min(1. / 2000.)
    difference = torch.where(available & (prior_index >= 0), difference, torch.zeros_like(difference))
    last_time = time_channels.gather(1, last_index.clamp_min(0))
    last_time = torch.where(last_index >= 0, last_time, torch.zeros_like(last_time))
    age = torch.where(valid, (time_channels - last_time).clamp_min(0.), torch.zeros_like(values))
    count = available.sum(dim=1).to(values.dtype)
    seen = count > 0
    first_index = torch.where(available, positions, torch.full_like(positions, length)).amin(dim=1)
    final_index = last_index[:, -1]
    first = values.gather(1, first_index.clamp_max(length - 1).unsqueeze(1)).squeeze(1)
    last = values.gather(1, final_index.clamp_min(0).unsqueeze(1)).squeeze(1)
    first_t = time_channels.gather(1, first_index.clamp_max(length - 1).unsqueeze(1)).squeeze(1)
    last_t = time_channels.gather(1, final_index.clamp_min(0).unsqueeze(1)).squeeze(1)
    first, last = (torch.where(seen, v, torch.zeros_like(v)) for v in (first, last))
    last_t = torch.where(seen, last_t, torch.zeros_like(last_t))
    mean = values.sum(dim=1) / count.clamp_min(1.)
    change = last - first
    slope = change * (60. / 2000.) / (last_t - first_t).clamp_min(1. / 2000.)
    fraction = count / valid.sum(dim=1).to(values.dtype).clamp_min(1.)
    age_at_window = (window - last_t).clamp_min(0.)
    summary = torch.cat((last, mean, change, slope, fraction, age_at_window, seen.to(values.dtype)), dim=-1)
    return difference, age, summary


def observed_features(x, signal_min, signal_span):
    span = torch.where(signal_span > 0, signal_span, torch.ones_like(signal_span))
    valid = torch.isfinite(x[..., 68:69]) & (x[..., 68:69] > .5)
    available = valid & (x[..., 34:68] > .5)
    values = torch.where(available, x[..., :34], torch.zeros_like(x[..., :34]))
    times = torch.where(valid, x[..., 69:70], torch.zeros_like(x[..., 69:70]))
    window = torch.where(valid, x[..., 70:71], torch.zeros_like(x[..., 70:71])).amax(dim=1)
    difference, age, signal_summary = _history(values, available, times, valid, window)
    coefficients, requirements, offsets, divisors = _physical_constants(x.device, x.dtype)
    raw = values * span + signal_min
    physical_available = available.to(x.dtype) @ requirements == requirements.sum(dim=0)
    physical = (raw @ coefficients + offsets) / divisors
    physical = torch.where(physical_available, physical, torch.zeros_like(physical))
    _, physical_age, physical_summary = _history(physical, physical_available, times, valid, window)
    sequence = torch.cat((values, difference, available.to(x.dtype), age, times,
                          window.unsqueeze(1).expand(-1, x.shape[1], -1),
                          physical, physical_available.to(x.dtype), physical_age), dim=-1)
    sequence = torch.where(valid, sequence, torch.zeros_like(sequence))
    events = torch.cat((values[..., EVENT_INDICES], difference[..., EVENT_INDICES],
                        available[..., EVENT_INDICES].to(x.dtype), age[..., EVENT_INDICES]), dim=-1)
    summary = torch.cat((signal_summary, physical_summary, window, times.amax(dim=1)), dim=-1)
    return sequence, valid, summary, events, window


def _masked_mean(values, valid):
    return torch.where(valid, values, torch.zeros_like(values)).sum(dim=1) / valid.sum(dim=1).clamp_min(1).to(values.dtype)


def _last_valid(values, valid):
    positions = torch.arange(values.shape[1], device=values.device).view(1, -1)
    indices = torch.where(valid.squeeze(-1), positions, 0).amax(dim=1)
    last = values.gather(1, indices[:, None, None].expand(-1, 1, values.shape[-1])).squeeze(1)
    return torch.where(valid.any(dim=1), last, torch.zeros_like(last))


def _normalized_cumulative(weights):
    cumulative = torch.cumsum(weights, dim=1)
    total = cumulative[:, -1:]
    return cumulative / torch.where(total > 0, total, torch.ones_like(total))


class ICLNet(nn.Module):
    def __init__(self, width, depth, dropout, decoder, signal_min, signal_span):
        super().__init__()
        self.decoder = decoder
        self.register_buffer("signal_min", torch.as_tensor(signal_min, dtype=torch.float32).detach().clone())
        self.register_buffer("signal_span", torch.as_tensor(signal_span, dtype=torch.float32).detach().clone())
        self.register_buffer("feature_schema_sha256",
                             torch.tensor(list(bytes.fromhex(FEATURE_SCHEMA_SHA256)), dtype=torch.uint8))
        self.sequence_projection = nn.Linear(180, width)
        self.event_encoder = nn.Sequential(nn.Linear(28, width), nn.SiLU(), nn.Linear(width, width))
        self.event_gate = nn.Linear(width, width)
        self.input_norm = nn.LayerNorm(width)
        self.input_dropout = nn.Dropout(dropout)
        self.encoder = nn.GRU(width, width, num_layers=depth, batch_first=True, dropout=dropout if depth > 1 else 0.0)
        self.attention = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.summary_projection = nn.Sequential(nn.LayerNorm(338), nn.Linear(338, width), nn.SiLU())
        self.context_projection = nn.Sequential(
            nn.LayerNorm(5 * width), nn.Linear(5 * width, 2 * width), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * width, width), nn.SiLU())
        mass_outputs = 2 if decoder == "recurrent_segmented" else 1
        rate_outputs = {"recurrent_direct": N_INTERVALS, "recurrent_spline": N_SPLINE_KNOTS,
                        "recurrent_segmented": 2 * N_INTERVALS}[decoder]
        self.mass_head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, mass_outputs))
        self.rate_head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, rate_outputs))
        times = torch.tensor(OUTPUT_TIMES.tolist(), dtype=torch.float32)
        self.register_buffer("output_times", times)
        self.register_buffer("interval_widths", torch.diff(times).view(1, -1) / 2000.0)
        centers = 0.5 * (times[:-1] + times[1:])
        self.register_buffer("interval_centers", centers)
        if decoder == "recurrent_spline":
            knots = torch.linspace(float(centers[0]), float(centers[-1]), N_SPLINE_KNOTS)
            left = torch.searchsorted(knots, centers, right=True).sub(1).clamp(0, N_SPLINE_KNOTS - 2)
            fraction = (centers - knots[left]) / (knots[left + 1] - knots[left])
            interpolation = torch.zeros(N_INTERVALS, N_SPLINE_KNOTS)
            interpolation.scatter_(1, left[:, None], (1 - fraction)[:, None])
            interpolation.scatter_add_(1, (left + 1)[:, None], fraction[:, None])
            self.register_buffer("rate_knot_times", knots)
            self.register_buffer("rate_interpolation", interpolation)

    def forward(self, x):
        sequence, valid, summary, event_sequence, window = observed_features(x, self.signal_min, self.signal_span)
        events = self.event_encoder(event_sequence)
        events = torch.where(valid, events, torch.zeros_like(events))
        sequence = self.sequence_projection(sequence)
        sequence = sequence + torch.sigmoid(self.event_gate(sequence)) * events
        sequence = self.input_dropout(F.silu(self.input_norm(sequence)))
        sequence = torch.where(valid, sequence, torch.zeros_like(sequence))
        encoded, _ = self.encoder(sequence)
        encoded = torch.where(valid, encoded, torch.zeros_like(encoded))
        logits = self.attention(encoded).masked_fill(~valid, -1e4)
        attention = torch.softmax(logits, dim=1) * valid.to(encoded.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = (attention * encoded).sum(dim=1)
        context = self.context_projection(torch.cat((
            _last_valid(encoded, valid), _masked_mean(encoded, valid), pooled,
            _masked_mean(events, valid), self.summary_projection(summary)), dim=1))
        masses = F.softplus(self.mass_head(context))
        rates = F.softplus(self.rate_head(context)) + 1e-8
        if self.decoder == "recurrent_segmented":
            boundary = window.clamp(0.0, 1.0) * 2000.0
            past_dt = (torch.minimum(self.output_times[1:][None, :], boundary)
                       - torch.minimum(self.output_times[:-1][None, :], boundary)) / 2000.0
            future_dt = (self.interval_widths - past_dt).clamp_min(0.0)
            past_rates, future_rates = rates.split(N_INTERVALS, dim=1)
            past_mass = masses[:, 0] * past_dt.sum(dim=1).gt(0).to(masses.dtype)
            future_gain = masses[:, 1] * future_dt.sum(dim=1).gt(0).to(masses.dtype)
            cumulative = (_normalized_cumulative(past_rates * past_dt) * past_mass[:, None]
                          + _normalized_cumulative(future_rates * future_dt) * future_gain[:, None])
        else:
            if self.decoder == "recurrent_spline":
                rates = rates @ self.rate_interpolation.T
            cumulative = _normalized_cumulative(rates * self.interval_widths) * masses[:, :1]
        return torch.cat((torch.zeros_like(masses[:, :1]), cumulative), dim=1)


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def seed_everything(seed, device):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def phase_schedule(meta, seed=SPLIT_SEED):
    groups = {}
    for row, (name, w, phase, _, _) in enumerate(meta):
        groups.setdefault((name, w), []).append((phase, row))
    rows, offsets = [], []
    for name, w in sorted(groups):
        members = np.array([row for _, row in sorted(groups[(name, w)])], dtype=np.int64)
        digest = hashlib.sha256(f"{seed}|{name}|{float(w):.12g}".encode()).digest()
        rows.append(members)
        offsets.append(int.from_bytes(digest[:8], "little") % len(members))
    return rows, offsets


def future_weights(windows):
    t = torch.tensor(OUTPUT_TIMES.tolist(), dtype=torch.float64)
    starts = torch.as_tensor(np.asarray(windows), dtype=torch.float64) * 60.0
    intervals = (t[:-1][None, :] >= starts[:, None]) * torch.diff(t)[None, :] / (2000.0 - starts[:, None])
    weights = torch.zeros((len(starts), 35), dtype=torch.float64)
    weights[:, :-1] += 0.5 * intervals
    weights[:, 1:] += 0.5 * intervals
    return weights.to(torch.float32)


def state_weights(windows):
    t = torch.tensor(OUTPUT_TIMES.tolist(), dtype=torch.float64)
    stops = torch.as_tensor(np.asarray(windows), dtype=torch.float64) * 60.0
    anchors = stops[:, None] == t[None, :]
    intervals = (t[1:][None, :] <= stops[:, None]) * torch.diff(t)[None, :] / stops[:, None]
    observed = torch.zeros((len(stops), 35), dtype=torch.float64)
    observed[:, :-1] += 0.5 * intervals
    observed[:, 1:] += 0.5 * intervals
    return anchors.to(torch.float32), observed.to(torch.float32)


def icl_loss(prediction, target, fw, anchor_w, observed_w, mode, beta):
    error = (prediction - target).abs()
    trajectory = (error * fw).sum(dim=1).mean()
    endpoint = error[:, -1].mean()
    base = 0.5 * (endpoint + trajectory)
    base = base + FULL_CURVE_WEIGHT * error.mean()
    absolute_error = (prediction - target).abs()
    anchor = (absolute_error * anchor_w).sum(dim=1).mean()
    observed = (absolute_error * observed_w).sum(dim=1).mean()
    auxiliary = 0.75 * anchor + 0.25 * observed
    if mode == "state_anchor_shape":
        predicted_shape = prediction / prediction[:, -1:].clamp_min(1e-6)
        target_shape = target / target[:, -1:].clamp_min(1e-6)
        shape = ((predicted_shape - target_shape).abs() * fw).sum(dim=1)
        shape = torch.where(target[:, -1] > 1e-6, shape, torch.zeros_like(shape)).mean()
        auxiliary = auxiliary + 0.25 * shape
    return base + float(beta) * auxiliary


def trapezoid(values, x):
    return (0.5 * (values[..., 1:] + values[..., :-1]) * np.diff(x)).sum(axis=-1)


def future_mae(target, prediction, windows):
    times = OUTPUT_TIMES.astype(np.float64)
    target, prediction, windows = np.asarray(target, float), np.asarray(prediction, float), np.asarray(windows)
    result = np.empty(len(target))
    for w in np.unique(windows):
        rows = windows == w
        start = int(np.searchsorted(times, 60 * w))
        error = np.abs(prediction[rows, start:] - target[rows, start:])
        result[rows] = trapezoid(error, times[start:]) / (2000 - 60 * w)
    return result


def validation_score(target, prediction, windows):
    future = future_mae(target, prediction, windows)
    endpoint = np.abs(np.asarray(prediction)[:, -1] - np.asarray(target)[:, -1])
    return float(np.mean([np.mean(.5 * endpoint[windows == w] + .5 * future[windows == w]) for w in np.unique(windows)]))


def predict(model, X, device):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(X), PREDICT_BATCH_SIZE):
            outputs.append(model(torch.from_numpy(np.ascontiguousarray(X[start:start + PREDICT_BATCH_SIZE])).to(device)))
    return torch.cat(outputs).cpu().numpy()


def train_network(hp, X, y, meta, scaler, epochs, seed, device, validation=None, report=None, label=""):
    seed_everything(seed, device)
    generator = np.random.default_rng(seed)
    model = ICLNet(hp["width"], hp["depth"], hp["dropout"], hp["decoder"],
                   scaler.input_min.tolist(), scaler.scale.tolist()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(hp["learning_rate"]), weight_decay=float(hp["weight_decay"]))
    rows, offsets = phase_schedule(meta)
    windows = np.array([m[1] for m in meta])
    fw_all = future_weights(windows)
    anchor_all, observed_all = state_weights(windows)
    x_all, y_all = torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(np.ascontiguousarray(y))
    best_state, best_score, best_epoch, stale = None, None, 0, 0
    started = time.time()
    for epoch in range(1, int(epochs) + 1):
        selected = np.asarray([r[(o + epoch - 1) % len(r)] for r, o in zip(rows, offsets)], dtype=np.int64)
        index = torch.from_numpy(selected[generator.permutation(len(selected))])
        xe, ye = x_all[index].to(device), y_all[index].to(device)
        fe, ae, oe = fw_all[index].to(device), anchor_all[index].to(device), observed_all[index].to(device)
        model.train()
        total = torch.zeros((), device=device)
        for s in range(0, len(index), BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            loss = icl_loss(model(xe[s:s + BATCH_SIZE]), ye[s:s + BATCH_SIZE], fe[s:s + BATCH_SIZE],
                            ae[s:s + BATCH_SIZE], oe[s:s + BATCH_SIZE], hp["auxiliary_mode"], hp["auxiliary_weight"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP, error_if_nonfinite=False)
            optimizer.step()
            total = total + loss.detach() * len(xe[s:s + BATCH_SIZE])
        train_loss = float((total / len(index)).cpu())
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
        score = None
        if validation is not None:
            vx, vy, vw = validation
            score = validation_score(vy * scaler.target_scale, predict(model, vx, device) * scaler.target_scale, vw)
            if best_score is None or score < best_score:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        elif epoch == epochs:
            best_epoch, best_state = epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            extra = f"  val J {score:.1f} kg (best {best_score:.1f} at {best_epoch})" if score is not None else ""
            print(f"  {label} epoch {epoch:3d}/{epochs}  loss {train_loss:.6f}{extra}  {time.time() - started:.0f}s", flush=True)
        if report is not None:
            report(epoch, score)
        if validation is not None and epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model.eval(), best_epoch, best_score


def search_fold(db, fold_map, fold, device, trials):
    import optuna
    roles = fold_roles(db, fold_map, fold)
    scaler = fit_scaler(db, roles["train"])
    X, y, meta = build_samples(db, roles["train"], scaler, phases=range(PHASES), training=True)
    vx, vy, vmeta = build_samples(db, roles["val"], scaler, windows=TUNING_WINDOWS_MIN)
    vw = np.array([m[1] for m in vmeta])
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEARCH["sampler_seed"], n_startup_trials=SEARCH["startup_trials"]),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=MIN_EPOCHS))
    for width, depth, rate, dropout, beta in ((128, 1, 7e-4, 0.0, 0.05), (192, 2, 3e-4, 0.1, 0.15)):
        for decoder in SEARCH["decoder"]:
            for mode in SEARCH["auxiliary_mode"]:
                study.enqueue_trial(dict(learning_rate=rate, weight_decay=1e-5, width=width, dropout=dropout,
                                         depth=depth, candidate_id=decoder, auxiliary_mode=mode, auxiliary_weight=beta))

    def objective(trial):
        hp = dict(
            learning_rate=trial.suggest_float("learning_rate", *SEARCH["learning_rate"], log=True),
            weight_decay=trial.suggest_categorical("weight_decay", list(SEARCH["weight_decay"])),
            width=trial.suggest_categorical("width", list(SEARCH["width"])),
            dropout=trial.suggest_categorical("dropout", list(SEARCH["dropout"])),
            depth=trial.suggest_categorical("depth", list(SEARCH["depth"])),
            decoder=trial.suggest_categorical("candidate_id", list(SEARCH["decoder"])),
            auxiliary_mode=trial.suggest_categorical("auxiliary_mode", list(SEARCH["auxiliary_mode"])),
            auxiliary_weight=trial.suggest_float("auxiliary_weight", *SEARCH["auxiliary_weight"], log=True))

        def report(epoch, score):
            trial.report(score, step=epoch)
            if epoch >= MIN_EPOCHS and trial.should_prune():
                raise optuna.TrialPruned()

        _, best_epoch, best_score = train_network(hp, X, y, meta, scaler, MAX_EPOCHS, 42, device,
                                                  validation=(vx, vy, vw), report=report,
                                                  label=f"fold {fold} trial {trial.number}")
        trial.set_user_attr("best_epoch", best_epoch)
        return best_score

    study.optimize(objective, n_trials=trials)
    best = study.best_trial
    hp = dict(width=best.params["width"], depth=best.params["depth"], dropout=best.params["dropout"],
              decoder=best.params["candidate_id"], learning_rate=best.params["learning_rate"],
              weight_decay=best.params["weight_decay"], auxiliary_mode=best.params["auxiliary_mode"],
              auxiliary_weight=best.params["auxiliary_weight"], epochs=int(best.user_attrs["best_epoch"]))
    return hp, float(best.value), study.trials_dataframe()


def conformal_radii(db, case_ids, scaler, models, device):
    index = {n: i for i, n in enumerate(db["case_ids"])}
    rank = math.ceil((len(case_ids) + 1) * COVERAGE)
    radii = {str(w): {} for w in WINDOWS_MIN}
    for condition in CONDITIONS:
        X, _, meta = build_samples(db, case_ids, scaler, condition=condition)
        prediction = np.mean([np.asarray(predict(m, X, device), np.float64) * scaler.target_scale for m in models], axis=0)
        target = db["targets"][[index[m[0]] for m in meta], -1]
        error = np.abs(target - prediction[:, -1])
        for w in WINDOWS_MIN:
            scores = {}
            for k, m in enumerate(meta):
                if m[1] == w:
                    scores[m[0]] = max(scores.get(m[0], 0.0), float(error[k]))
            radii[str(w)][condition] = float(np.sort(list(scores.values()))[rank - 1])
    return radii


def write_config(output, data_hash, folds, radii):
    config = dict(
        model="ICL-Net",
        description="Intervention-aware cumulative leakage network (state-anchored recurrent ICL, v7): "
                    "five cross-validation folds x three training seeds",
        dataset_sha256=data_hash, split_seed=SPLIT_SEED, seeds=list(SEEDS), windows_min=list(WINDOWS_MIN),
        output_times_s=OUTPUT_TIMES.tolist(), conditions=list(CONDITIONS),
        sensor_groups=[dict(id=f"group_{i}", name=n, channels=list(c)) for i, (n, c) in enumerate(GROUPS)],
        channels=[dict(index=i, column=c, name=n, unit=u) for i, (c, n, u) in enumerate(CHANNELS)],
        folds=sorted(folds.values(), key=lambda f: f["fold"]),
        conformal=dict(nominal_coverage=COVERAGE, calibration_cases_per_fold=30, order_statistic=28,
                       score="largest absolute endpoint residual over the fault repetitions of a calibration case",
                       interval="[max(0, M_hat(T) - q), max(0, M_hat(T) + q)]",
                       q_kg={str(k): radii[k] for k in sorted(radii)}))
    (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train ICL-Net (5 cross-validation folds x 3 seeds) and calibrate "
                                                 "its split-conformal endpoint intervals.")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "models"))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)), choices=range(5))
    parser.add_argument("--search", action="store_true",
                        help="re-run the 48-trial hyperparameter search of each fold (Optuna) before the final fits")
    parser.add_argument("--trials", type=int, default=SEARCH["trials"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    seed_everything(SEARCH["sampler_seed"], device)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    db = load_database(args.data_dir)
    status = "matches the paper" if db["data_hash"] == DATASET_SHA256 else "differs from the paper"
    print(f"dataset: {len(db['case_ids'])} transients, sha256 {db['data_hash'][:16]}... ({status}); device {device}")
    fold_map = make_folds(db)
    folds, radii = {}, {}
    for fold in args.folds:
        roles = fold_roles(db, fold_map, fold)
        if args.search:
            print(f"fold {fold}: hyperparameter search ({args.trials} trials)")
            hp, validation_j, trials = search_fold(db, fold_map, fold, device, args.trials)
            trials.to_csv(output / f"search_trials_fold{fold}.csv", index=False)
        else:
            hp, validation_j = dict(PAPER_HYPERPARAMETERS[fold]), None
        print(f"fold {fold}: {hp}")
        scaler = fit_scaler(db, roles["fit"])
        X, y, meta = build_samples(db, roles["fit"], scaler, phases=range(PHASES), training=True)
        models, states = [], []
        for seed in SEEDS:
            model, _, _ = train_network(hp, X, y, meta, scaler, hp["epochs"], seed, device, label=f"fold {fold} seed {seed}")
            models.append(model)
            states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        del X, y
        radii[fold] = conformal_radii(db, roles["cal"], scaler, models, device)
        architecture = {k: hp[k] for k in ("width", "depth", "dropout", "decoder")}
        training = {k: hp[k] for k in ("learning_rate", "weight_decay", "auxiliary_mode", "auxiliary_weight", "epochs")}
        target = output / f"icl_net_fold{fold}.pt"
        torch.save(dict(format="icl-net/v7", fold=fold, seeds=list(SEEDS), architecture=architecture,
                        training=training, scaler=scaler.to_dict(), state_dicts=states), target)
        folds[fold] = dict(fold=fold, file=target.name, **architecture, **training,
                           **({} if validation_j is None else dict(validation_J_kg=validation_j)),
                           parameters_per_network=sum(p.numel() for p in models[0].parameters()),
                           sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        write_config(output, db["data_hash"], folds, radii)
        print(f"fold {fold}: saved {target}")
    print(f"done: {output}")


if __name__ == "__main__":
    main()
