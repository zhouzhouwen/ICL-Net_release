import argparse
import hashlib
import itertools
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATASET_SHA256 = "7bcd8cf5530e8752455b374e6d723eb947457323d4f203ecea8cdf89ec46a070"
FEATURE_SCHEMA_SHA256 = "eb492045af18887e21abb22441519bab6682341bfded71a0dc25903102bce7c5"
SPLIT_SEED = 20260929
WINDOWS_MIN = (5, 8, 10, 11, 12, 13, 14, 15, 20)
PRIMARY_WINDOWS_MIN = (10, 12, 15)
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
COLUMNS = ("neutron_flux_pct", "reactor_trip", "si_signal", "t_cold_a_degc", "t_hot_a_degc", "t_avg_a_degc",
           "t_avg_b_degc", "t_hot_b_degc", "t_cold_b_degc", "p_rcs_b_pa", "prv_open", "p_prz_pa", "l_prz_pct",
           "charging_a_kgs", "charging_b_kgs", "letdown_kgs", "si_a_kgs", "si_b_kgs", "p_sg_a_pa", "p_sg_b_pa",
           "l_sg_a_pct", "l_sg_b_pct", "msiv_a_s", "msiv_b_s", "msrv_a", "msrv_b", "mfw_a_kgs", "mdafw_a_kgs",
           "tdafw_a_kgs", "ms_a_kgs", "mfw_b_kgs", "mdafw_b_kgs", "tdafw_b_kgs", "ms_b_kgs")
PREDICT_BATCH_SIZE = 256
COVERAGE = 0.9
PAPER = dict(
    clean_primary_J_kg=770.7235094453366,
    clean15_endpoint_mae_kg=472.0805244392689,
    clean15_endpoint_mape_pct=1.3748120759015232,
    degraded_primary_J_kg=872.3602628203855,
    inadmissible_forecasts=0,
    pair_delta_mae_kg=1268.451458627599,
    aggregate_coverage=0.9130864197530864,
    aggregate_width_kg=5317.516809893175,
    w_star_2t_conditions=12,
    w_star_2t_mean_min=12.0,
)


def _hash(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _rng(*parts):
    return np.random.default_rng(int(_hash(*parts)[:16], 16))


def load_database(data_dir):
    root = Path(data_dir).resolve()
    label_path = root / "spds.plt"
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
        inputs.append(a[:, INPUT_COLUMNS].astype(np.float32))
        targets.append(np.r_[0.0, np.cumsum(positive[1:], dtype=np.float64)][OUTPUT_TIMES])
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
    roles = dict(test=(fold,), cal=((fold + 1) % 5,))
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


def fault_spec(case_id, condition, repeat=0, seed=SPLIT_SEED):
    if condition == "clean":
        return [], -1, -1
    if condition.startswith("group_"):
        return list(GROUPS[int(condition.split("_")[1])][1]), 0, 2001
    n = int(condition[4:]) if condition.startswith("drop") else 7
    channels = sorted(_rng(seed, "fault", case_id, condition, repeat).choice(34, n, replace=False).tolist())
    start, end = (300, 420) if condition == "block7" else ((600, 2001) if condition == "outage7" else (0, 2001))
    return channels, start, end


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


def build_samples(db, case_ids, scaler, condition, seed=SPLIT_SEED):
    index = {n: i for i, n in enumerate(db["case_ids"])}
    specs = [(index[name], name, w, r) for name in case_ids for w in WINDOWS_MIN
             for r in range(10 if condition in RANDOM_CONDITIONS else 1)]
    X = np.empty((len(specs), 21, 71), dtype=np.float32)
    meta = []
    for j, (i, name, w, rep) in enumerate(specs):
        times = np.arange(0, w * 60 + 1, 60, dtype=np.int64)
        channels, start, end = fault_spec(name, condition, rep, seed)
        available = np.ones((len(times), 34), dtype=bool)
        if channels:
            available[np.ix_((times >= start) & (times < end), channels)] = False
        X[j] = observation_tensor(db["inputs"][i][times], times, w, scaler, available)
        meta.append((name, w, rep))
    return X, meta


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
    device = torch.device(name)
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    return device


def load_models(models_dir, device):
    models_dir = Path(models_dir).resolve()
    config = json.loads((models_dir / "config.json").read_text(encoding="utf-8"))
    folds = []
    for entry in config["folds"]:
        record = torch.load(models_dir / entry["file"], map_location="cpu", weights_only=True)
        arch = record["architecture"]
        networks = []
        for state in record["state_dicts"]:
            if bytes(state["feature_schema_sha256"].tolist()).hex() != FEATURE_SCHEMA_SHA256:
                raise ValueError(f"{entry['file']}: feature schema differs from this script")
            network = ICLNet(arch["width"], arch["depth"], arch["dropout"], arch["decoder"],
                             state["signal_min"], state["signal_span"])
            network.load_state_dict(state)
            networks.append(network.to(device).eval())
        folds.append(dict(fold=int(record["fold"]), scaler=Scaler(**record["scaler"]), networks=networks,
                          q=config["conformal"]["q_kg"][str(record["fold"])]))
    return config, folds


def ensemble_predict(fold, X, device):
    outputs = []
    with torch.inference_mode():
        for network in fold["networks"]:
            parts = [network(torch.from_numpy(np.ascontiguousarray(X[s:s + PREDICT_BATCH_SIZE])).to(device))
                     for s in range(0, len(X), PREDICT_BATCH_SIZE)]
            outputs.append(torch.cat(parts).cpu().numpy().astype(np.float64) * fold["scaler"].target_scale)
    return np.mean(outputs, axis=0)


def trapezoid(values, x):
    return (0.5 * (values[..., 1:] + values[..., :-1]) * np.diff(x)).sum(axis=-1)


def future_mae(target, prediction, windows):
    times = OUTPUT_TIMES.astype(np.float64)
    result = np.empty(len(target))
    for w in np.unique(windows):
        rows = windows == w
        start = int(np.searchsorted(times, 60 * w))
        result[rows] = trapezoid(np.abs(prediction[rows, start:] - target[rows, start:]), times[start:]) / (2000 - 60 * w)
    return result


def evaluate(db, folds, device, output):
    fold_map = make_folds(db)
    index = {n: i for i, n in enumerate(db["case_ids"])}
    rows, clean_curves = [], {}
    started = time.time()
    for fold in folds:
        roles = fold_roles(db, fold_map, fold["fold"])
        for role, key in (("test", "test"), ("calibration", "cal")):
            for condition in CONDITIONS:
                X, meta = build_samples(db, roles[key], fold["scaler"], condition)
                prediction = ensemble_predict(fold, X, device)
                target = db["targets"][[index[m[0]] for m in meta]]
                windows = np.array([m[1] for m in meta])
                future = future_mae(target, prediction, windows)
                for k, (name, w, rep) in enumerate(meta):
                    rows.append((fold["fold"], role, name, w, condition, rep, target[k, -1], prediction[k, -1], future[k]))
                    if role == "test" and condition == "clean":
                        clean_curves[(name, w)] = (prediction[k], target[k], X[k])
        print(f"  fold {fold['fold']}: predictions for 60 transients x 9 windows x 15 conditions "
              f"({time.time() - started:.0f}s)", flush=True)
    df = pd.DataFrame(rows, columns=["fold", "role", "case_id", "window_min", "condition", "repeat",
                                     "target_end_kg", "pred_end_kg", "future_mae_kg"])
    df["abs_error_kg"] = (df.pred_end_kg - df.target_end_kg).abs()
    calibration = df[df.role == "calibration"]
    scores = calibration.groupby(["fold", "window_min", "condition", "case_id"]).abs_error_kg.max()
    q = scores.groupby(["fold", "window_min", "condition"]).apply(
        lambda s: np.sort(s.to_numpy())[math.ceil((len(s) + 1) * COVERAGE) - 1]).rename("q_kg")
    df = df.merge(q.reset_index(), on=["fold", "window_min", "condition"], how="left")
    df["lower_kg"] = np.maximum(0.0, df.pred_end_kg - df.q_kg)
    df["upper_kg"] = np.maximum(0.0, df.pred_end_kg + df.q_kg)
    df["covered"] = (df.target_end_kg >= df.lower_kg) & (df.target_end_kg <= df.upper_kg)
    stored_q = max(abs(fold["q"][str(w)][c] - float(q.loc[(fold["fold"], w, c)]))
                   for fold in folds for w in WINDOWS_MIN for c in CONDITIONS)
    test = df[df.role == "test"].copy()
    test["ape_pct"] = 100 * test.abs_error_kg / test.target_end_kg
    test["width_kg"] = test.upper_kg - test.lower_kg
    cells = test.groupby(["window_min", "condition", "case_id"], as_index=False).agg(
        endpoint_mae_kg=("abs_error_kg", "mean"), endpoint_mse_kg2=("abs_error_kg", lambda e: float((e ** 2).mean())),
        endpoint_mape_pct=("ape_pct", "mean"), future_mae_kg=("future_mae_kg", "mean"),
        worst_error_kg=("abs_error_kg", "max"), covered=("covered", "all"), width_kg=("width_kg", "mean"))
    cells["J_kg"] = 0.5 * (cells.endpoint_mae_kg + cells.future_mae_kg)
    table = cells.groupby(["window_min", "condition"], as_index=False).agg(
        n_cases=("case_id", "size"), endpoint_mae_kg=("endpoint_mae_kg", "mean"),
        endpoint_rmse_kg=("endpoint_mse_kg2", lambda v: float(np.sqrt(v.mean()))),
        endpoint_mape_pct=("endpoint_mape_pct", "mean"), future_mae_kg=("future_mae_kg", "mean"),
        J_kg=("J_kg", "mean"), coverage=("covered", "mean"), mean_interval_width_kg=("width_kg", "mean"))
    primary = cells[cells.window_min.isin(PRIMARY_WINDOWS_MIN)]
    clean_j = primary[primary.condition == "clean"].groupby("case_id").J_kg.mean().mean()
    degraded_j = primary[primary.condition != "clean"].groupby("case_id").J_kg.mean().mean()
    clean15 = table[(table.window_min == 15) & (table.condition == "clean")].iloc[0]
    curves = np.stack([c[0] for c in clean_curves.values()])
    inadmissible = int(((np.abs(curves[:, 0]) > 1e-3) | (curves.min(axis=1) < -1e-3)
                        | (np.diff(curves, axis=1).min(axis=1) < -1e-3)).sum())
    pair_scores = []
    meta = [r for r in db["meta"] if not r["is_extra"]]
    for key in sorted({(r["location"], r["core"], r["isolation_min"]) for r in meta}):
        early = next(r["case_id"] for r in meta if (r["location"], r["core"], r["isolation_min"]) == key and r["relief_min"] == 6)
        late = next(r["case_id"] for r in meta if (r["location"], r["core"], r["isolation_min"]) == key and r["relief_min"] == 10)
        errors = []
        for w in PRIMARY_WINDOWS_MIN:
            if (early, w) not in clean_curves or (late, w) not in clean_curves:
                continue
            (pe, te, xe), (pl, tl, xl) = clean_curves[(early, w)], clean_curves[(late, w)]
            if not np.array_equal(xe, xl):
                errors.append(abs((pl[-1] - pe[-1]) - (tl[-1] - te[-1])))
        if errors:
            pair_scores.append(np.mean(errors))
    worst = test.groupby(["condition", "window_min", "case_id"]).abs_error_kg.max().reset_index()
    w_star = {}
    for condition, part in worst.groupby("condition"):
        passing = part.assign(ok=part.abs_error_kg <= 2000.0).groupby("window_min").ok.mean()
        w_star[condition] = next((float(w) for w in sorted(passing.index) if (passing.loc[w:] >= 0.9).all()), None)
    achieved = [v for v in w_star.values() if v is not None]
    result = dict(
        clean_primary_J_kg=float(clean_j),
        clean15_endpoint_mae_kg=float(clean15.endpoint_mae_kg),
        clean15_endpoint_mape_pct=float(clean15.endpoint_mape_pct),
        degraded_primary_J_kg=float(degraded_j),
        inadmissible_forecasts=inadmissible,
        clean_forecasts_checked=len(curves),
        pair_delta_mae_kg=float(np.mean(pair_scores)),
        operation_pairs=len(pair_scores),
        aggregate_coverage=float(table.coverage.mean()),
        aggregate_width_kg=float(table.mean_interval_width_kg.mean()),
        cells_reaching_90pct=int((table.coverage >= 0.9).sum()),
        cells=len(table),
        w_star_2t_conditions=len(achieved),
        w_star_2t_mean_min=float(np.mean(achieved)) if achieved else None,
        w_star_2t_by_condition=w_star,
        max_abs_difference_from_released_q_kg=float(stored_q),
        test_transients=int(test.case_id.nunique()),
        folds=[f["fold"] for f in folds],
    )
    output.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["abs_error_kg"]).to_csv(output / "predictions.csv", index=False)
    table.to_csv(output / "metrics_by_window_condition.csv", index=False)
    (output / "summary.json").write_text(json.dumps(dict(reproduced=result, paper=PAPER), indent=2) + "\n", encoding="utf-8")
    return result


def print_report(result, complete):
    def line(label, value, paper, fmt):
        reference = fmt.format(paper) if complete else "-"
        print(f"  {label:<50}{fmt.format(value):>14}{reference:>14}")
    print(f"\n  {'':<50}{'reproduced':>14}{'paper':>14}")
    line("Clean signals: primary J, W = 10/12/15 min (kg)", result["clean_primary_J_kg"], PAPER["clean_primary_J_kg"], "{:.2f}")
    line("Clean signals: endpoint MAE at W = 15 min (kg)", result["clean15_endpoint_mae_kg"], PAPER["clean15_endpoint_mae_kg"], "{:.2f}")
    line("Clean signals: endpoint MAPE at W = 15 min (%)", result["clean15_endpoint_mape_pct"], PAPER["clean15_endpoint_mape_pct"], "{:.3f}")
    line("Sensor loss: primary J, 14 conditions (kg)", result["degraded_primary_J_kg"], PAPER["degraded_primary_J_kg"], "{:.2f}")
    line("Physically inadmissible clean forecasts", result["inadmissible_forecasts"], PAPER["inadmissible_forecasts"], "{:d}")
    line("Cooldown-timing pair error (kg)", result["pair_delta_mae_kg"], PAPER["pair_delta_mae_kg"], "{:.2f}")
    line("Interval coverage, nominal 90% (%)", 100 * result["aggregate_coverage"], 100 * PAPER["aggregate_coverage"], "{:.2f}")
    line("Mean 90% interval width (kg)", result["aggregate_width_kg"], PAPER["aggregate_width_kg"], "{:.2f}")
    line("Conditions accepted within 2 t (W* found)", result["w_star_2t_conditions"], PAPER["w_star_2t_conditions"], "{:d}")
    print(f"\n  forecasts checked for admissibility: {result['clean_forecasts_checked']}; "
          f"operation pairs: {result['operation_pairs']}; cells at >= 90% coverage: "
          f"{result['cells_reaching_90pct']} of {result['cells']}")
    print(f"  max |recomputed q - released q|: {result['max_abs_difference_from_released_q_kg']:.3g} kg")


def read_windows(path):
    frame = pd.read_csv(path)
    missing = [c for c in ("sample_id", "time_s", *COLUMNS) if c not in frame.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}. See data/README.md for the input format.")
    samples = []
    for sample_id, part in frame.groupby("sample_id", sort=False):
        part = part.sort_values("time_s")
        times = part.time_s.to_numpy(dtype=float)
        if not np.all(times == np.round(times)) or times[0] != 0 or np.any(np.diff(times) <= 0) or len(times) > 21:
            raise SystemExit(f"sample {sample_id}: time_s must be increasing integers starting at 0 (at most 21 rows)")
        times = times.astype(np.int64)
        if "window_min" in part and part.window_min.notna().any():
            window = float(part.window_min.dropna().iloc[0])
        else:
            window = times[-1] / 60.0
        if window not in WINDOWS_MIN or times[-1] > window * 60:
            raise SystemExit(f"sample {sample_id}: the observation window must be one of {WINDOWS_MIN} min "
                             f"and contain the observed times")
        values = part[list(COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        truth = None
        if "true_endpoint_kg" in part and part.true_endpoint_kg.notna().any():
            truth = float(part.true_endpoint_kg.dropna().iloc[0])
        samples.append(dict(sample_id=str(sample_id), times=times, values=values, window=int(window), truth=truth))
    return samples


def sensor_condition(times, available):
    missing = ~available
    if not missing.any():
        return "clean", True
    channels = np.flatnonzero(missing.any(axis=0))
    if missing[:, channels].all():
        for i, (_, members) in enumerate(GROUPS):
            if set(channels.tolist()) == set(members):
                return f"group_{i}", True
        if len(channels) in (1, 3, 7):
            return f"drop{len(channels)}", True
    if len(channels) == 7:
        mask = np.isin(np.arange(34), channels)
        if np.array_equal(missing, np.outer((times >= 300) & (times < 420), mask)):
            return "block7", True
        if np.array_equal(missing, np.outer(times >= 600, mask)):
            return "outage7", True
    return "other", False


def predict_windows(samples, folds, device):
    available = [np.isfinite(s["values"]) for s in samples]
    conditions = [sensor_condition(s["times"], a) for s, a in zip(samples, available)]
    fold_curves = []
    for fold in folds:
        X = np.stack([observation_tensor(s["values"], s["times"], s["window"], fold["scaler"], a)
                      for s, a in zip(samples, available)])
        fold_curves.append(ensemble_predict(fold, X, device))
    results = []
    for k, sample in enumerate(samples):
        condition, calibrated = conditions[k]
        endpoints, lowers, uppers = [], [], []
        for fold, curves in zip(folds, fold_curves):
            radii = fold["q"][str(sample["window"])]
            q = radii[condition] if calibrated else max(v for c, v in radii.items() if c != "clean")
            endpoints.append(curves[k, -1])
            lowers.append(max(0.0, curves[k, -1] - q))
            uppers.append(max(0.0, curves[k, -1] + q))
        curve = np.mean([curves[k] for curves in fold_curves], axis=0)
        row = dict(sample_id=sample["sample_id"], window_min=sample["window"], observations=len(sample["times"]),
                   sensor_condition=condition,
                   interval_basis="calibrated" if calibrated else "conservative (pattern outside the 15 evaluated conditions)",
                   pred_endpoint_kg=curve[-1], pi90_lower_kg=min(lowers), pi90_upper_kg=max(uppers),
                   pi90_width_kg=max(uppers) - min(lowers),
                   fold_endpoint_min_kg=min(endpoints), fold_endpoint_max_kg=max(endpoints))
        row.update({f"pred_M_{t}s_kg": v for t, v in zip(OUTPUT_TIMES.tolist(), curve)})
        if sample["truth"] is not None:
            error = curve[-1] - sample["truth"]
            row.update(true_endpoint_kg=sample["truth"], error_kg=error, abs_error_kg=abs(error),
                       ape_pct=100 * abs(error) / sample["truth"] if sample["truth"] > 0 else float("nan"),
                       inside_pi90=bool(min(lowers) <= sample["truth"] <= max(uppers)))
        results.append(row)
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained ICL-Net models, or forecast the cumulative "
                                                 "leakage for new observation windows (--input).")
    parser.add_argument("--models", default=str(ROOT / "models"))
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--input", help="CSV of observation windows (format in data/README.md)")
    parser.add_argument("--output", help="output directory (evaluation) or CSV file (--input)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    config, folds = load_models(args.models, device)
    print(f"loaded {sum(len(f['networks']) for f in folds)} ICL-Net networks "
          f"(folds {[f['fold'] for f in folds]}) from {Path(args.models).resolve()} on {device}")
    if args.input:
        output = Path(args.output or ROOT / "outputs" / "prediction_results.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        results = predict_windows(read_windows(args.input), folds, device)
        results.to_csv(output, index=False)
        columns = ["sample_id", "window_min", "sensor_condition", "pred_endpoint_kg", "pi90_lower_kg", "pi90_upper_kg"]
        columns += [c for c in ("true_endpoint_kg", "abs_error_kg") if c in results]
        print(results[columns].round(1).to_string(index=False))
        print(f"\nwrote {len(results)} forecasts to {output}")
        return
    if not (Path(args.data_dir) / "spds.plt").is_file():
        print(f"\nDataset not found in {Path(args.data_dir).resolve()}.\n"
              "The simulation data are confidential (see data/README.md). Authorized users place the files in "
              "data/raw/.\nTo forecast your own observation windows with the released models, run:\n"
              "  python scripts/test.py --input <windows.csv>")
        sys.exit(2)
    db = load_database(args.data_dir)
    status = "matches the paper" if db["data_hash"] == DATASET_SHA256 else "differs from the paper"
    print(f"dataset: {len(db['case_ids'])} transients, sha256 {db['data_hash'][:16]}... ({status})")
    output = Path(args.output or ROOT / "outputs" / "test")
    result = evaluate(db, folds, device, output)
    complete = len(folds) == 5 and db["data_hash"] == DATASET_SHA256
    print_report(result, complete)
    print(f"\nwrote predictions.csv, metrics_by_window_condition.csv and summary.json to {output.resolve()}")


if __name__ == "__main__":
    main()
