"""
tpms_generator.py
Moduł geometryczny TPMS (Gyroid) oraz syntetyczny homogenizator sztywności (Synthetic Homogenization Engine).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import scipy.linalg


def gyroid_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nx: float,
    ny: float,
    nz: float,
    t: float,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
) -> np.ndarray:
    """
    Oblicza pole skalarne Gyroidu w przestrzeni 3D:
    f(x, y, z) = sin(kx * x)*cos(ky * y) + sin(ky * y)*cos(kz * z) + sin(kz * z)*cos(kx * x) - t
    """
    kx = 2.0 * np.pi * nx / Lx
    ky = 2.0 * np.pi * ny / Ly
    kz = 2.0 * np.pi * nz / Lz

    term1 = np.sin(kx * x) * np.cos(ky * y)
    term2 = np.sin(ky * y) * np.cos(kz * z)
    term3 = np.sin(kz * z) * np.cos(kx * x)

    return term1 + term2 + term3 - t


def diamond_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nx: float,
    ny: float,
    nz: float,
    t: float,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
) -> np.ndarray:
    """
    Oblicza pole skalarne powierzchni TPMS Schwarz Diamond (D-type) w przestrzeni 3D:
    f(x, y, z) = sin(kx*x)*sin(ky*y)*sin(kz*z) + sin(kx*x)*cos(ky*y)*cos(kz*z) +
                 cos(kx*x)*sin(ky*y)*cos(kz*z) + cos(kx*x)*cos(ky*y)*sin(kz*z) - t
    """
    kx = 2.0 * np.pi * nx / Lx
    ky = 2.0 * np.pi * ny / Ly
    kz = 2.0 * np.pi * nz / Lz

    X = kx * x
    Y = ky * y
    Z = kz * z

    term1 = np.sin(X) * np.sin(Y) * np.sin(Z)
    term2 = np.sin(X) * np.cos(Y) * np.cos(Z)
    term3 = np.cos(X) * np.sin(Y) * np.cos(Z)
    term4 = np.cos(X) * np.cos(Y) * np.sin(Z)

    return term1 + term2 + term3 + term4 - t


def primitive_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nx: float,
    ny: float,
    nz: float,
    t: float,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
) -> np.ndarray:
    """
    Oblicza pole skalarne powierzchni TPMS Schwarz Primitive (P-type) w przestrzeni 3D:
    f(x, y, z) = cos(kx * x) + cos(ky * y) + cos(kz * z) - t
    """
    kx = 2.0 * np.pi * nx / Lx
    ky = 2.0 * np.pi * ny / Ly
    kz = 2.0 * np.pi * nz / Lz

    return np.cos(kx * x) + np.cos(ky * y) + np.cos(kz * z) - t


def auxetic_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nx: float,
    ny: float,
    nz: float,
    t: float,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    reentrant_factor: float = 0.35,
) -> np.ndarray:
    """
    Oblicza pole skalarne komórki metamateriału auksetycznego (Re-entrant Star 3D, nu < 0):
    f(x, y, z) = cos(kx*x)*cos(ky*y)*cos(kz*z) - c*(cos(2*kx*x) + cos(2*ky*y) + cos(2*kz*z)) - t
    Wprowadza ujemny współczynnik Poissona dzięki geometrii wklęsłych krawędzi (re-entrant angles).
    """
    kx = 2.0 * np.pi * nx / Lx
    ky = 2.0 * np.pi * ny / Ly
    kz = 2.0 * np.pi * nz / Lz

    X = kx * x
    Y = ky * y
    Z = kz * z

    base = np.cos(X) * np.cos(Y) * np.cos(Z)
    reentrant = reentrant_factor * (np.cos(2.0 * X) + np.cos(2.0 * Y) + np.cos(2.0 * Z))
    return base - reentrant - t


def neovius_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nx: float,
    ny: float,
    nz: float,
    t: float,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
) -> np.ndarray:
    """
    Oblicza pole skalarne powierzchni TPMS Neovius (C(P)-type) w przestrzeni 3D:
    f(x, y, z) = [3*(cos(kx*x) + cos(ky*y) + cos(kz*z)) + 4*cos(kx*x)*cos(ky*y)*cos(kz*z)] / 3.0 - t
    Grupa Im-3m, komórka z 12 diagonalnymi kanałami krzyżowymi.
    """
    kx = 2.0 * np.pi * nx / Lx
    ky = 2.0 * np.pi * ny / Ly
    kz = 2.0 * np.pi * nz / Lz

    X = kx * x
    Y = ky * y
    Z = kz * z

    val = 3.0 * (np.cos(X) + np.cos(Y) + np.cos(Z)) + 4.0 * np.cos(X) * np.cos(Y) * np.cos(Z)
    return (val / 3.0) - t


def iwp_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nx: float,
    ny: float,
    nz: float,
    t: float,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
) -> np.ndarray:
    """
    Oblicza pole skalarne Schoen I-WP (I-Wrapped Package) w przestrzeni 3D:
    f(x, y, z) = 2*(cos(X)*cos(Y) + cos(Y)*cos(Z) + cos(Z)*cos(X)) - (cos(2X) + cos(2Y) + cos(2Z)) - t
    Dwie asymetryczne fazy labiryntu o skrajnie różnej objętości (wymienniki dwupłynowe, membrany).
    """
    kx = 2.0 * np.pi * nx / Lx
    ky = 2.0 * np.pi * ny / Ly
    kz = 2.0 * np.pi * nz / Lz

    X = kx * x
    Y = ky * y
    Z = kz * z

    term1 = 2.0 * (np.cos(X) * np.cos(Y) + np.cos(Y) * np.cos(Z) + np.cos(Z) * np.cos(X))
    term2 = np.cos(2.0 * X) + np.cos(2.0 * Y) + np.cos(2.0 * Z)
    return term1 - term2 - t


def octet_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nx: float,
    ny: float,
    nz: float,
    t: float,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
) -> np.ndarray:
    """
    Oblicza pole skalarne Schoen F-RD / Octet-Truss w przestrzeni 3D:
    f(x, y, z) = 4*cos(X)*cos(Y)*cos(Z) - (cos(2X)*cos(2Y) + cos(2Y)*cos(2Z) + cos(2Z)*cos(2X)) - t
    Krata oktaedryczno-tetraedryczna w układzie FCC (Fm-3m), czysty mechanizm rozciągania.
    """
    kx = 2.0 * np.pi * nx / Lx
    ky = 2.0 * np.pi * ny / Ly
    kz = 2.0 * np.pi * nz / Lz

    X = kx * x
    Y = ky * y
    Z = kz * z

    term1 = 4.0 * np.cos(X) * np.cos(Y) * np.cos(Z)
    term2 = np.cos(2.0 * X) * np.cos(2.0 * Y) + np.cos(2.0 * Y) * np.cos(2.0 * Z) + np.cos(2.0 * Z) * np.cos(2.0 * X)
    return term1 - term2 - t


def generate_tpms_grid(
    tpms_type: str,
    nx: float,
    ny: float,
    nz: float,
    t: float,
    resolution: int = 35,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    mode: str = "network",
    wall_thickness: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tworzy siatkę 3D i wartości pola skalarnego wybranego typu TPMS / metamateriału:
    - 'gyroid' (Schoen G)
    - 'diamond' (Schwarz D)
    - 'primitive' (Schwarz P)
    - 'auxetic' (Re-entrant Auxetic nu < 0)
    - 'neovius' (Schoen C(P), 12 kanałów diagonalnych)
    - 'iwp' (Schoen I-WP, komory asymetryczne)
    - 'octet' (Schoen F-RD / Octet-Truss)

    Wspiera tryb morfologii:
    - 'network': lity szkielet jednofazowy F = f(x, y, z)
    - 'sheet': cienkościenna podwójna powłoka F = |f(x, y, z)| - wall_thickness
    """
    x = np.linspace(0, Lx, resolution)
    y = np.linspace(0, Ly, resolution)
    z = np.linspace(0, Lz, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    type_str = str(tpms_type).lower()
    if "diamond" in type_str:
        F_raw = diamond_field(X, Y, Z, nx, ny, nz, t, Lx, Ly, Lz)
    elif "primitive" in type_str:
        F_raw = primitive_field(X, Y, Z, nx, ny, nz, t, Lx, Ly, Lz)
    elif "auxetic" in type_str or "auksetyk" in type_str:
        F_raw = auxetic_field(X, Y, Z, nx, ny, nz, t, Lx, Ly, Lz)
    elif "neovius" in type_str:
        F_raw = neovius_field(X, Y, Z, nx, ny, nz, t, Lx, Ly, Lz)
    elif "iwp" in type_str or "i-wp" in type_str:
        F_raw = iwp_field(X, Y, Z, nx, ny, nz, t, Lx, Ly, Lz)
    elif "octet" in type_str or "frd" in type_str or "f-rd" in type_str:
        F_raw = octet_field(X, Y, Z, nx, ny, nz, t, Lx, Ly, Lz)
    else:
        F_raw = gyroid_field(X, Y, Z, nx, ny, nz, t, Lx, Ly, Lz)

    if str(mode).lower() == "sheet":
        # W trybie Sheet poziom zero to granica ścianki o grubości 2*wall_thickness
        F = np.abs(F_raw) - float(wall_thickness)
    else:
        F = F_raw

    return X, Y, Z, F


def generate_gyroid_grid(
    nx: float,
    ny: float,
    nz: float,
    t: float,
    resolution: int = 35,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    mode: str = "network",
    wall_thickness: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tworzy siatkę 3D Gyroidu."""
    return generate_tpms_grid("gyroid", nx, ny, nz, t, resolution, Lx, Ly, Lz, mode=mode, wall_thickness=wall_thickness)


def generate_diamond_grid(
    nx: float,
    ny: float,
    nz: float,
    t: float,
    resolution: int = 35,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    mode: str = "network",
    wall_thickness: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tworzy siatkę 3D Schwarz Diamond."""
    return generate_tpms_grid("diamond", nx, ny, nz, t, resolution, Lx, Ly, Lz, mode=mode, wall_thickness=wall_thickness)


def generate_primitive_grid(
    nx: float,
    ny: float,
    nz: float,
    t: float,
    resolution: int = 35,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    mode: str = "network",
    wall_thickness: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tworzy siatkę 3D Schwarz Primitive."""
    return generate_tpms_grid("primitive", nx, ny, nz, t, resolution, Lx, Ly, Lz, mode=mode, wall_thickness=wall_thickness)


def generate_auxetic_grid(
    nx: float,
    ny: float,
    nz: float,
    t: float,
    resolution: int = 35,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    mode: str = "network",
    wall_thickness: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tworzy siatkę 3D Re-entrant Auxetic."""
    return generate_tpms_grid("auxetic", nx, ny, nz, t, resolution, Lx, Ly, Lz, mode=mode, wall_thickness=wall_thickness)


def generate_neovius_grid(
    nx: float,
    ny: float,
    nz: float,
    t: float,
    resolution: int = 35,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    mode: str = "network",
    wall_thickness: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tworzy siatkę 3D Neovius TPMS."""
    return generate_tpms_grid("neovius", nx, ny, nz, t, resolution, Lx, Ly, Lz, mode=mode, wall_thickness=wall_thickness)


def generate_iwp_grid(
    nx: float,
    ny: float,
    nz: float,
    t: float,
    resolution: int = 35,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    mode: str = "network",
    wall_thickness: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tworzy siatkę 3D Schoen I-WP."""
    return generate_tpms_grid("iwp", nx, ny, nz, t, resolution, Lx, Ly, Lz, mode=mode, wall_thickness=wall_thickness)


def generate_octet_grid(
    nx: float,
    ny: float,
    nz: float,
    t: float,
    resolution: int = 35,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    mode: str = "network",
    wall_thickness: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tworzy siatkę 3D Octet-Truss / Schoen F-RD."""
    return generate_tpms_grid("octet", nx, ny, nz, t, resolution, Lx, Ly, Lz, mode=mode, wall_thickness=wall_thickness)


def compute_relative_density(
    nx: float,
    ny: float,
    nz: float,
    t: float,
    resolution: int = 30,
    tpms_type: str = "gyroid",
    mode: str = "network",
    wall_thickness: float = 0.20,
) -> float:
    """
    Oblicza numeryczną gęstość względną (udział objętościowy materiału)
    dla obszaru, w którym F(x, y, z) <= 0.
    """
    _, _, _, F = generate_tpms_grid(
        tpms_type=tpms_type,
        nx=nx,
        ny=ny,
        nz=nz,
        t=t,
        resolution=resolution,
        mode=mode,
        wall_thickness=wall_thickness,
    )
    rho = float(np.mean(F <= 0))
    # Zabezpieczenie przed skrajnymi wartościami numerycznymi
    return float(np.clip(rho, 0.05, 0.95))


def matrix_to_vector_21(C: np.ndarray) -> np.ndarray:
    """
    Konwertuje symetryczną macierz 6x6 do 21-elementowego wektora
    reprezentującego górny trójkąt (wiersz po wierszu).
    Indeksy:
    (0,0), (0,1), ..., (0,5),
    (1,1), ..., (1,5),
    ...
    (5,5)
    """
    indices = np.triu_indices(6)
    return C[indices]


def vector_21_to_matrix(vec: np.ndarray) -> np.ndarray:
    """
    Rekonstruuje pełną symetryczną macierz 6x6 z 21 elementów górnego trójkąta.
    Obsługuje zarówno pojedynczy wektor (21,), jak i batch (N, 21).
    """
    if vec.ndim == 1:
        C = np.zeros((6, 6), dtype=np.float64)
        indices = np.triu_indices(6)
        C[indices] = vec
        C = C + C.T - np.diag(np.diag(C))
        return C
    elif vec.ndim == 2:
        N = vec.shape[0]
        C = np.zeros((N, 6, 6), dtype=np.float64)
        indices = np.triu_indices(6)
        for i in range(N):
            C[i][indices] = vec[i]
            C[i] = C[i] + C[i].T - np.diag(np.diag(C[i]))
        return C
    else:
        raise ValueError(f"Nieobsługiwany wymiar wektora: {vec.ndim}")


def synthetic_stiffness_matrix(
    nx: float,
    ny: float,
    nz: float,
    t: float,
    E0: float = 100.0,
    nu0: float = 0.28,
    noise_level: float = 0.015,
    rng: Optional[np.random.Generator] = None,
    tpms_type: str = "gyroid",
    mode: str = "network",
    wall_thickness: float = 0.20,
) -> np.ndarray:
    """
    Fizycznie uzasadniony generator syntetycznej ortotropowej macierzy sztywności 6x6 (w GPa).
    Obsługuje 4 geometrie:
    - Gyroid: struktura chiralna, m ~ 1.85, gamma_shear ~ 0.36
    - Diamond: struktura węzłowa tetraedryczna, wyższa sztywność osiowa m ~ 1.76, gamma_shear ~ 0.41
    - Primitive: regularna sieć ortogonalna, wysoka anizotropia osiowa m ~ 1.95, gamma_shear ~ 0.28
    - Auxetic: komórka wklęsła re-entrant, ujemny współczynnik Poissona nu < 0, S12 > 0, C12 < 0

    Oraz 2 tryby morfologiczne:
    - network: lity szkielet jednofazowy (bending-dominated)
    - sheet: cienkościenna podwójna powłoka (stretching-dominated, ~2x wyższa sztywność, m ~ 1.20)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    type_str = str(tpms_type).lower()
    is_diamond = "diamond" in type_str
    is_primitive = "primitive" in type_str
    is_auxetic = ("auxetic" in type_str) or ("auksetyk" in type_str)
    is_neovius = "neovius" in type_str
    is_iwp = "iwp" in type_str or "i-wp" in type_str
    is_octet = "octet" in type_str or "frd" in type_str or "f-rd" in type_str
    is_sheet = str(mode).lower() == "sheet"

    # 1. Gęstość względna rho
    if is_sheet:
        rho = float(np.clip(1.8 * float(wall_thickness) * (1.0 - 0.15 * (t ** 2)), 0.06, 0.94))
    else:
        if is_diamond:
            rho = 0.5 + 0.38 * np.tanh(1.1 * t)
        elif is_primitive:
            rho = 0.5 + 0.36 * np.tanh(1.0 * t)
        elif is_auxetic:
            rho = 0.5 + 0.32 * np.tanh(1.1 * t)
        elif is_neovius:
            rho = 0.5 + 0.35 * np.tanh(1.1 * t)
        elif is_iwp:
            rho = 0.5 + 0.33 * np.tanh(1.15 * t)
        elif is_octet:
            rho = 0.5 + 0.36 * np.tanh(1.1 * t)
        else:
            rho = 0.5 + 0.35 * np.tanh(1.2 * t)
        rho = float(np.clip(rho, 0.08, 0.92))

    # 2. Wykładnik Gibsona-Ashby'ego dla TPMS i trybu
    if is_diamond:
        m = 1.76
        E_base = E0 * 1.08
        alpha = 1.22
        gamma_shear = 0.41
        nu_base = 0.30
    elif is_primitive:
        m = 1.95
        E_base = E0 * 0.95
        alpha = 1.28
        gamma_shear = 0.28
        nu_base = 0.25
    elif is_auxetic:
        m = 1.80
        E_base = E0 * 0.92
        alpha = 1.18
        gamma_shear = 0.38
        nu_base = -0.22  # Ujemny współczynnik Poissona!
    elif is_neovius:
        m = 1.80
        E_base = E0 * 1.02
        alpha = 1.20
        gamma_shear = 0.44  # Wysoki moduł ścinania ze względu na 12 kanałów diagonalnych
        nu_base = 0.29
    elif is_iwp:
        m = 1.74
        E_base = E0 * 1.05
        alpha = 1.12
        gamma_shear = 0.38
        nu_base = 0.28
    elif is_octet:
        m = 1.55  # Stretching-dominated truss mechanics
        E_base = E0 * 1.20  # Najwyższa baza sztywności
        alpha = 1.14
        gamma_shear = 0.46
        nu_base = 0.31
    else:  # Gyroid
        m = 1.85
        E_base = E0
        alpha = 1.15
        gamma_shear = 0.36
        nu_base = nu0

    # Przejście w tryb Sheet (stretching-dominated deformation mechanics)
    if is_sheet:
        m = 1.10 if is_octet else 1.20
        E_base = E_base * 1.85  # Zbliżenie do granicy Hashina-Shtrikmana
        gamma_shear = gamma_shear * 1.15

    E_iso = E_base * (rho ** m)

    # 3. Anizotropia komórki
    n_mean = (nx * ny * nz) ** (1.0 / 3.0)
    rx = nx / n_mean
    ry = ny / n_mean
    rz = nz / n_mean

    Ex = E_iso * (rx ** alpha)
    Ey = E_iso * (ry ** alpha)
    Ez = E_iso * (rz ** alpha)

    # 4. Współczynniki Poissona
    if is_auxetic:
        # Auksetyk: nu < 0 (rozszerzanie poprzeczne)
        nu_xy = nu_base * (1.0 - 0.2 * (1.0 - rho)) * np.sqrt(rx / ry)
        nu_yx = nu_xy * (Ey / Ex)
        nu_xz = nu_base * (1.0 - 0.2 * (1.0 - rho)) * np.sqrt(rx / rz)
        nu_zx = nu_xz * (Ez / Ex)
        nu_yz = nu_base * (1.0 - 0.2 * (1.0 - rho)) * np.sqrt(ry / rz)
        nu_zy = nu_yz * (Ez / Ey)

        # Zabezpieczenie fizyczne na ujemny współczynnik Poissona
        nu_xy = np.clip(nu_xy, -0.45, -0.05)
        nu_yx = np.clip(nu_yx, -0.45, -0.05)
        nu_xz = np.clip(nu_xz, -0.45, -0.05)
        nu_zx = np.clip(nu_zx, -0.45, -0.05)
        nu_yz = np.clip(nu_yz, -0.45, -0.05)
        nu_zy = np.clip(nu_zy, -0.45, -0.05)

        # Moduły ścinania dla auksetyka
        Gyz = gamma_shear * np.sqrt(Ey * Ez) / (1.0 - np.sqrt(abs(nu_yz * nu_zy)))
        Gxz = gamma_shear * np.sqrt(Ex * Ez) / (1.0 - np.sqrt(abs(nu_xz * nu_zx)))
        Gxy = gamma_shear * np.sqrt(Ex * Ey) / (1.0 - np.sqrt(abs(nu_xy * nu_yx)))
    else:
        nu_xy = nu_base * (1.0 - 0.2 * (1.0 - rho)) * np.sqrt(rx / ry)
        nu_yx = nu_xy * (Ey / Ex)
        nu_xz = nu_base * (1.0 - 0.2 * (1.0 - rho)) * np.sqrt(rx / rz)
        nu_zx = nu_xz * (Ez / Ex)
        nu_yz = nu_base * (1.0 - 0.2 * (1.0 - rho)) * np.sqrt(ry / rz)
        nu_zy = nu_yz * (Ez / Ey)

        max_nu = 0.45
        nu_xy = np.clip(nu_xy, 0.05, max_nu)
        nu_yx = np.clip(nu_yx, 0.05, max_nu)
        nu_xz = np.clip(nu_xz, 0.05, max_nu)
        nu_zx = np.clip(nu_zx, 0.05, max_nu)
        nu_yz = np.clip(nu_yz, 0.05, max_nu)
        nu_zy = np.clip(nu_zy, 0.05, max_nu)

        Gyz = gamma_shear * np.sqrt(Ey * Ez) / (1.0 + np.sqrt(nu_yz * nu_zy))
        Gxz = gamma_shear * np.sqrt(Ex * Ez) / (1.0 + np.sqrt(nu_xz * nu_zx))
        Gxy = gamma_shear * np.sqrt(Ex * Ey) / (1.0 + np.sqrt(nu_xy * nu_yx))

    # Budujemy macierz podatności S (Compliance Matrix) 6x6
    S = np.zeros((6, 6), dtype=np.float64)
    S[0, 0] = 1.0 / Ex
    S[1, 1] = 1.0 / Ey
    S[2, 2] = 1.0 / Ez

    S[0, 1] = S[1, 0] = -nu_yx / Ey
    S[0, 2] = S[2, 0] = -nu_zx / Ez
    S[1, 2] = S[2, 1] = -nu_zy / Ez

    S[3, 3] = 1.0 / Gyz
    S[4, 4] = 1.0 / Gxz
    S[5, 5] = 1.0 / Gxy

    # Odwracamy do macierzy sztywności C
    try:
        C = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        C = np.linalg.pinv(S)

    # Symetryzacja numeryczna
    C = 0.5 * (C + C.T)

    # Dodanie fizycznego szumu losowego z zachowaniem symetrii i dodatniej określoności
    if noise_level > 0.0:
        raw_noise = rng.normal(0.0, noise_level, (6, 6))
        sym_noise = 0.5 * (raw_noise + raw_noise.T) * np.mean(np.diag(C))
        C_noisy = C + sym_noise

        # Gwarancja dodatniej określoności (eigenvalue projection)
        eigvals, eigvecs = np.linalg.eigh(C_noisy)
        min_eig = np.min(np.diag(C)) * 0.05
        eigvals_clipped = np.maximum(eigvals, min_eig)
        C = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    # Ostateczna symetryzacja
    C = 0.5 * (C + C.T)
    return C


def generate_synthetic_dataset(
    num_samples: int = 600,
    random_seed: int = 42,
    noise_level: float = 0.015,
    tpms_type: str = "gyroid",
    mode: str = "network",
    wall_thickness: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generuje syntetyczny zbiór danych treningowych dla wybranego typu TPMS i trybu morfologii:
    X: wektory [nx, ny, nz, t] o kształcie (N, 4)
    y: wektory 21 elementów górnego trójkąta macierzy C o kształcie (N, 21)
    """
    rng = np.random.default_rng(random_seed)

    # Równomierny lub quasi-losowy rozkład parametrów w zadanych granicach
    nx = rng.uniform(1.0, 5.0, num_samples)
    ny = rng.uniform(1.0, 5.0, num_samples)
    nz = rng.uniform(1.0, 5.0, num_samples)
    t = rng.uniform(-0.95, 0.95, num_samples)

    X = np.column_stack([nx, ny, nz, t])
    y_list = []

    for i in range(num_samples):
        C_i = synthetic_stiffness_matrix(
            nx[i], ny[i], nz[i], t[i],
            noise_level=noise_level,
            rng=rng,
            tpms_type=tpms_type,
            mode=mode,
            wall_thickness=wall_thickness,
        )
        vec_21 = matrix_to_vector_21(C_i)
        y_list.append(vec_21)

    y = np.array(y_list, dtype=np.float32)
    X = X.astype(np.float32)
    return X, y


def compliance_tensor_from_matrix(S_mat: np.ndarray) -> np.ndarray:
    """
    Konwertuje macierz podatności 6x6 (w konwencji inżynierskiej Voigta)
    do pełnego tensora podatności czwartego rzędu S_ijkl (3x3x3x3).
    """
    voigt_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    S_4th = np.zeros((3, 3, 3, 3), dtype=np.float64)

    for p in range(6):
        i, j = voigt_map[p]
        factor_p = 0.5 if p >= 3 else 1.0
        for q in range(6):
            k, l = voigt_map[q]
            factor_q = 0.5 if q >= 3 else 1.0
            val = S_mat[p, q] * factor_p * factor_q

            # Wypełniamy symetrie tensora S_ijkl
            S_4th[i, j, k, l] = val
            S_4th[j, i, k, l] = val
            S_4th[i, j, l, k] = val
            S_4th[j, i, l, k] = val
            S_4th[k, l, i, j] = val
            S_4th[l, k, i, j] = val
            S_4th[k, l, j, i] = val
            S_4th[l, k, j, i] = val

    return S_4th


def calculate_directional_youngs_modulus(
    C: np.ndarray,
    num_theta: int = 50,
    num_phi: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Oblicza kierunkowy moduł Younga E(u) w przestrzeni 3D na podstawie macierzy sztywności C (6x6).
    1/E(u) = S_ijkl * u_i * u_j * u_k * u_l
    Zwraca siatki współrzędnych 3D (X, Y, Z) oraz wartości E_mod.
    """
    # Sprawdzamy symetrię i odwracalność
    C_sym = 0.5 * (C + C.T)
    try:
        S = np.linalg.inv(C_sym)
    except np.linalg.LinAlgError:
        S = np.linalg.pinv(C_sym)

    S_4th = compliance_tensor_from_matrix(S)

    # Kąty sferyczne: theta in [0, pi], phi in [0, 2*pi]
    theta = np.linspace(0, np.pi, num_theta)
    phi = np.linspace(0, 2.0 * np.pi, num_phi)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

    # Wektor jednostkowy kierunku u = [lx, ly, lz]
    lx = np.sin(THETA) * np.cos(PHI)
    ly = np.sin(THETA) * np.sin(PHI)
    lz = np.cos(THETA)

    # Obliczenie wektorowe 1/E(u) = S_ijkl * u_i * u_j * u_k * u_l
    U = np.stack([lx, ly, lz], axis=-1)  # shape: (num_theta, num_phi, 3)

    # Einstein summation: S_4th ma ksztalt (3,3,3,3), U ma ksztalt (num_theta, num_phi, 3)
    inv_E = np.einsum("ijkl,abi,abj,abk,abl->ab", S_4th, U, U, U, U)

    # Zabezpieczenie przed dzieleniem przez zero lub wartosciami ujemnymi
    inv_E = np.maximum(inv_E, 1e-9)
    E_mod = 1.0 / inv_E

    # Współrzędne bryły polarnej w przestrzeni 3D
    X_surf = E_mod * lx
    Y_surf = E_mod * ly
    Z_surf = E_mod * lz

    return X_surf, Y_surf, Z_surf, E_mod
