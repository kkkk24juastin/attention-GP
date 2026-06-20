import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from config import (
    BODY_LENGTH_SCALE,
    FEATURE_NAMES,
    INNER_DIAMETER_SCALE,
    NOSE_SHAPE_NAME,
    NOSE_SHAPE_PARAMETER_COLUMN,
    NOSE_LENGTH_SCALE,
    NOSE_THICKNESS_SCALE,
    OPENROCKET_JAR_FILE,
    OPENROCKET_ORK_TEMPLATE,
    OPENROCKET_RUNNER_FILE,
    OPENROCKET_SIMULATION_INDEX,
    OPENROCKET_TIMEOUT_SECONDS,
    OUTER_DIAMETER_SCALE,
    nose_shape_parameter_from_length,
)


RESULT_PREFIX = "OPENROCKET_RESULT\t"


def _first_child(component, tag):
    child = component.find(tag)
    if child is None:
        raise ValueError(f"OpenRocket template missing <{tag}> under <{component.tag}>")
    return child


def _component_by_name(root, tag, name):
    for component in root.iter(tag):
        if component.findtext("name") == name:
            return component
    raise ValueError(f"OpenRocket template missing <{tag}> named {name!r}")


def _set_text(component, tag, value):
    _first_child(component, tag).text = f"{float(value):.12g}"


def design_values_from_row(row, prefix="ga_x"):
    values = []
    for idx, feature in enumerate(FEATURE_NAMES, start=1):
        prefixed_name = f"{prefix}{idx}" if prefix else None
        if prefixed_name and prefixed_name in row and pd.notna(row[prefixed_name]):
            values.append(float(row[prefixed_name]))
        elif feature in row and pd.notna(row[feature]):
            values.append(float(row[feature]))
        else:
            raise ValueError(f"Design row missing {prefixed_name} / {feature}")
    return values


def write_design_ork(design_values, output_path):
    x1, x2, x3, x4, x5 = [float(value) for value in design_values]
    nose_shape_parameter = nose_shape_parameter_from_length(x1)
    nose_length = x1 * NOSE_LENGTH_SCALE
    body_radius = 0.5 * x2 * OUTER_DIAMETER_SCALE
    nose_thickness = x3 * NOSE_THICKNESS_SCALE
    body_length = x4 * BODY_LENGTH_SCALE
    body_inner_radius = 0.5 * x5 * INNER_DIAMETER_SCALE
    body_thickness = max(body_radius - body_inner_radius, 1e-5)
    inner_outer_radius = min(0.0095, 0.98 * body_inner_radius)

    with ZipFile(OPENROCKET_ORK_TEMPLATE, "r") as source:
        xml_data = source.read("rocket.ork")
        root = ET.fromstring(xml_data)

        nose = _component_by_name(root, "nosecone", "Nose cone")
        body = _component_by_name(root, "bodytube", "Body tube")
        inner = _component_by_name(root, "innertube", "Inner Tube")

        _set_text(nose, "length", nose_length)
        _set_text(nose, "thickness", nose_thickness)
        _first_child(nose, "shape").text = NOSE_SHAPE_NAME
        _set_text(nose, "shapeparameter", nose_shape_parameter)
        _set_text(nose, "aftradius", body_radius)
        _set_text(nose, "aftshoulderradius", max(body_radius - nose_thickness, 1e-5))
        _set_text(body, "length", body_length)
        _first_child(body, "radius").text = f"auto {body_radius:.12g}"
        _set_text(body, "thickness", body_thickness)
        _set_text(inner, "outerradius", inner_outer_radius)

        output_path = Path(output_path)
        with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as target:
            for item in source.infolist():
                if item.filename == "rocket.ork":
                    target.writestr(
                        item,
                        ET.tostring(root, encoding="utf-8", xml_declaration=True),
                    )
                else:
                    target.writestr(item, source.read(item.filename))


def run_openrocket(ork_path):
    runtime_dir = Path(ork_path).parent / "openrocket_runtime"
    prefs_dir = runtime_dir / "prefs"
    home_dir = runtime_dir / "home"
    prefs_dir.mkdir(parents=True, exist_ok=True)
    home_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "java",
        "-Djava.awt.headless=true",
        f"-Djava.util.prefs.userRoot={prefs_dir}",
        f"-Duser.home={home_dir}",
        "--add-exports",
        "java.base/java.lang=ALL-UNNAMED",
        "--add-exports",
        "java.desktop/sun.awt=ALL-UNNAMED",
        "--add-exports",
        "java.desktop/sun.java2d=ALL-UNNAMED",
        "-cp",
        str(OPENROCKET_JAR_FILE),
        str(OPENROCKET_RUNNER_FILE),
        str(ork_path),
        str(OPENROCKET_SIMULATION_INDEX),
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        timeout=OPENROCKET_TIMEOUT_SECONDS,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        raise RuntimeError(output.strip())
    for line in output.splitlines():
        if line.startswith(RESULT_PREFIX):
            parts = line.split("\t")
            return {
                "最大飞行高度": float(parts[1]),
                "flight_time": float(parts[2]),
                "time_to_apogee": float(parts[3]),
                "max_velocity": float(parts[4]),
                "max_acceleration": float(parts[5]),
                "simulation_note": "ok",
            }
    raise RuntimeError(f"OpenRocket result line not found. Output:\n{output}")


def simulate_design(design_values, work_dir):
    if not OPENROCKET_JAR_FILE.exists():
        raise FileNotFoundError(f"OpenRocket jar 不存在: {OPENROCKET_JAR_FILE}")
    if not OPENROCKET_ORK_TEMPLATE.exists():
        raise FileNotFoundError(f"OpenRocket 模板不存在: {OPENROCKET_ORK_TEMPLATE}")

    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    ork_path = work_path / "candidate.ork"
    write_design_ork(design_values, ork_path)
    result = run_openrocket(ork_path)
    result[NOSE_SHAPE_PARAMETER_COLUMN] = nose_shape_parameter_from_length(design_values[0])
    return result


def simulate_design_row(row, work_dir, prefix="ga_x"):
    return simulate_design(design_values_from_row(row, prefix=prefix), work_dir)


def simulate_design_matrix(X, work_dir):
    X = np.asarray(X, dtype=float)
    responses = []
    for idx, design_values in enumerate(X, start=1):
        result = simulate_design(design_values, Path(work_dir) / f"design_{idx:04d}")
        responses.append(float(result["最大飞行高度"]))
    return np.asarray(responses, dtype=float)
