"""
export_3d.py
Moduł eksportu geometrii TPMS do formatów inżynierskich CAD i druku 3D:
- STL (Standard Triangle Language - binary / ASCII)
- STEP (ISO 10303-21 AP203/AP214 Faceted B-Rep)
- IGES (ANSI Y14.26M Initial Graphics Exchange Specification)
- OBJ (Wavefront 3D)
- PLY (Polygon File Format)
"""

import os
import tempfile
from typing import Tuple, Optional
import numpy as np
from skimage.measure import marching_cubes
import trimesh

import tpms_generator as tg

# Import OpenCASCADE for STEP and IGES export
try:
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    from OCP.gp import gp_Pnt
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IGESControl import IGESControl_Writer
    from OCP.IFSelect import IFSelect_RetDone
    OCP_AVAILABLE = True
except ImportError:
    OCP_AVAILABLE = False


def generate_tpms_mesh(
    tpms_type: str,
    nx: float,
    ny: float,
    nz: float,
    t: float,
    resolution: int = 24,
    size_x_mm: float = 20.0,
    size_y_mm: float = 20.0,
    size_z_mm: float = 20.0,
    mode: str = "network",
    wall_thickness: float = 0.20,
) -> trimesh.Trimesh:
    """
    Ekstrahuje siatkę trójkątów z niejawnego pola TPMS przy użyciu algorytmu Marching Cubes.
    Obsługuje 4 geometrie: Gyroid, Diamond, Primitive, Auxetic oraz tryby Network i Sheet.
    Skaluje współrzędne do fizycznych wymiarów w milimetrach.
    """
    # 1. Siatka skalarna
    X, Y, Z, F = tg.generate_tpms_grid(
        tpms_type=tpms_type,
        nx=nx,
        ny=ny,
        nz=nz,
        t=t,
        resolution=resolution,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        mode=mode,
        wall_thickness=wall_thickness,
    )

    # Spacing w milimetrach
    spacing = (
        size_x_mm / float(resolution - 1),
        size_y_mm / float(resolution - 1),
        size_z_mm / float(resolution - 1),
    )

    # 2. Algorytm Marching Cubes dla poziomu 0.0 z bezpiecznym fallbackiem
    try:
        verts, faces, normals, _ = marching_cubes(F, level=0.0, spacing=spacing)
    except ValueError:
        # Fallback gdy poziom 0.0 jest poza zakresem F
        f_min, f_max = float(np.min(F)), float(np.max(F))
        fallback_level = float(np.clip(0.0, f_min + 1e-4, f_max - 1e-4))
        try:
            verts, faces, normals, _ = marching_cubes(F, level=fallback_level, spacing=spacing)
        except Exception:
            verts = np.zeros((0, 3), dtype=np.float32)
            faces = np.zeros((0, 3), dtype=int)
            normals = np.zeros((0, 3), dtype=np.float32)

    # 3. Utworzenie obiektu Trimesh
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals,
        process=True,
    )
    return mesh


def export_to_stl(mesh: trimesh.Trimesh, binary: bool = True) -> bytes:
    """Eksportuje siatkę do formatu STL (domyślnie binarny)."""
    file_type = "stl" if binary else "stl_ascii"
    return mesh.export(file_type=file_type)


def export_to_obj(mesh: trimesh.Trimesh) -> bytes:
    """Eksportuje siatkę do formatu Wavefront OBJ."""
    return mesh.export(file_type="obj").encode("utf-8")


def export_to_ply(mesh: trimesh.Trimesh) -> bytes:
    """Eksportuje siatkę do formatu PLY."""
    return mesh.export(file_type="ply")


def export_to_step(mesh: trimesh.Trimesh) -> bytes:
    """
    Eksportuje siatkę do formatu STEP (ISO 10303-21) jako zbiór ścianek Faceted B-Rep
    z wykorzystaniem silnika OpenCASCADE (OCP).
    """
    if not OCP_AVAILABLE:
        raise RuntimeError("Biblioteka OpenCASCADE (OCP) nie jest dostępna w środowisku.")

    if len(mesh.faces) == 0:
        raise ValueError("Siatka nie zawiera trójkątów (brak izopowierzchni). Wybierz inne parametry geometrii.")

    pnts = [gp_Pnt(float(v[0]), float(v[1]), float(v[2])) for v in mesh.vertices]
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    for f in mesh.faces:
        try:
            p1, p2, p3 = pnts[f[0]], pnts[f[1]], pnts[f[2]]
            if (
                p1.SquareDistance(p2) < 1e-12
                or p2.SquareDistance(p3) < 1e-12
                or p3.SquareDistance(p1) < 1e-12
            ):
                continue
            poly = BRepBuilderAPI_MakePolygon(p1, p2, p3, True)
            if not poly.IsDone():
                continue
            face_maker = BRepBuilderAPI_MakeFace(poly.Wire())
            if face_maker.IsDone():
                builder.Add(compound, face_maker.Face())
        except Exception:
            continue

    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp:
        path = tmp.name

    try:
        writer = STEPControl_Writer()
        writer.Transfer(compound, STEPControl_AsIs)
        writer.Write(path)
        with open(path, "rb") as f_in:
            data = f_in.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)

    return data


def export_to_iges(mesh: trimesh.Trimesh) -> bytes:
    """
    Eksportuje siatkę do formatu IGES (ANSI Y14.26M) z wykorzystaniem silnika OpenCASCADE (OCP).
    """
    if not OCP_AVAILABLE:
        raise RuntimeError("Biblioteka OpenCASCADE (OCP) nie jest dostępna w środowisku.")

    if len(mesh.faces) == 0:
        raise ValueError("Siatka nie zawiera trójkątów (brak izopowierzchni). Wybierz inne parametry geometrii.")

    pnts = [gp_Pnt(float(v[0]), float(v[1]), float(v[2])) for v in mesh.vertices]
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    for f in mesh.faces:
        try:
            p1, p2, p3 = pnts[f[0]], pnts[f[1]], pnts[f[2]]
            if (
                p1.SquareDistance(p2) < 1e-12
                or p2.SquareDistance(p3) < 1e-12
                or p3.SquareDistance(p1) < 1e-12
            ):
                continue
            poly = BRepBuilderAPI_MakePolygon(p1, p2, p3, True)
            if not poly.IsDone():
                continue
            face_maker = BRepBuilderAPI_MakeFace(poly.Wire())
            if face_maker.IsDone():
                builder.Add(compound, face_maker.Face())
        except Exception:
            continue

    with tempfile.NamedTemporaryFile(suffix=".iges", delete=False) as tmp:
        path = tmp.name

    try:
        writer = IGESControl_Writer()
        writer.AddShape(compound)
        writer.ComputeModel()
        writer.Write(path)
        with open(path, "rb") as f_in:
            data = f_in.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)

    return data
