from enum import auto, unique
from typing import List, Tuple

from strenum import SnakeCaseStrEnum, UppercaseStrEnum

from vital.data.config import LabelEnum

PATIENT_ID_REGEX = r"\d{4}"
HDF5_FILENAME_PATTERN = "{patient_id}_{view}.h5"
IMG_FILENAME_PATTERN = "{patient_id}_{view}_{tag}{ext}"
IMG_FORMAT = ".nii.gz"
ATTRS_CACHE_FORMAT = "npz"
ATTRS_FILENAME_PATTERN = "{patient_id}{ext}"
TABULAR_ATTRS_FORMAT = "yaml"

IN_CHANNELS: int = 1
"""Number of input channels of the images in the dataset."""

DEFAULT_SIZE: Tuple[int, int] = (256, 256)
"""Default size at which the raw B-mode images are resized."""


@unique
class Label(LabelEnum):
    """Identifiers of the different anatomical structures available in the dataset's segmentation mask."""

    BG = 0
    """BackGround"""
    LV = 1
    """Left Ventricle"""
    MYO = 2
    """MYOcardium"""


@unique
class View(UppercaseStrEnum):
    """Names of the different views available for each patient."""

    A4C = auto()
    """Apical 4 Chamber"""
    A2C = auto()
    """Apical 2 Chamber"""
    A3C = auto()
    """Apical 3 Chamber"""


@unique
class TimeSeriesAttribute(SnakeCaseStrEnum):
    """Names of the attributes that are temporal sequences of values measured at each frame in the sequences."""

    gls = auto()
    ls_left = auto()
    ls_right = auto()
    """Longitudinal Strain (LS) of the endocardium: Global (GLS), left (LSL) and right (LSR)."""
    lv_area = auto()
    """Number of pixels covered by the left ventricle (LV)."""
    lv_length = auto()
    """Distance between the LV's base and apex."""
    myo_thickness_left = auto()
    myo_thickness_right = auto()
    """Average thickness of the myocardium (MYO) over left/right segments."""


@unique
class TabularAttribute(SnakeCaseStrEnum):
    """Name of the attributes that are scalar values extracted from the patient's record or the images."""

    ef = auto()
    """Ejection Fraction (EF)."""
    edv = auto()
    """End-Diastolic Volume (EDV)."""
    esv = auto()
    """End-Systolic Volume (ESV)."""
    a4c_ed_sc_min = "a4c_ed_sc_min"  # Assign the string manually because numbers are discarded by `auto`
    a4c_ed_sc_max = "a4c_ed_sc_max"  # ""
    a4c_ed_lc_min = "a4c_ed_lc_min"  # ""
    a4c_ed_lc_max = "a4c_ed_lc_max"  # ""
    a2c_ed_ic_min = "a2c_ed_ic_min"  # ""
    a2c_ed_ic_max = "a2c_ed_ic_max"  # ""
    a2c_ed_ac_min = "a2c_ed_ac_min"  # ""
    a2c_ed_ac_max = "a2c_ed_ac_max"  # ""
    """Peak concave(-) or convex(+) Curvatures of the Septal/Lateral Inferior/Anterior walls in A4C/A2C view, at ED."""
    age = auto()
    sex = auto()
    height = auto()
    weight = auto()
    bmi = auto()
    """Body Mass Index (BMI)."""
    hf = auto()
    """Heart Failure (HF)."""
    cad = auto()
    """Coronary Artery Disease (CAD)."""
    pad = auto()
    """Peripheral Artery Disease (PAD)."""
    stroke = auto()
    tobacco = auto()
    diabetes = auto()
    dyslipidemia = auto()
    etiology = auto()
    ddd = auto()
    """Cumulative Daily Defined Dose (DDD) of treatment given to the patient."""
    bradycardic = auto()
    """Whether the patient's treatment contains a bradycardic."""
    ace_inhibitor = auto()
    """Whether the patient's treatment contains an Angiotensin Enzyme Converter (ACE) inhibitor."""
    arb = auto()
    """Whether the patient's treatment contains an Angiotensin Receptor Blocker (ARB)."""
    tz_diuretic = auto()
    """Whether the patient's treatment contains a ThiaZide (TZ) diuretic."""
    central_acting = auto()
    """Whether the patient's treatment contains a central-acting (or centrally acting) agent."""
    beta_blocker = auto()
    """Whether the patient's treatment contains a beta blocker."""
    spironolactone = auto()
    """Whether the patient's treatment contains a Spironolactone."""
    alpha_blocker = auto()
    """Whether the patient's treatment contains an alpha blocker."""
    ccb = auto()
    """Whether the patient's treatment contains a Calcium Channel Blocker (CCB)."""
    sbp_24 = "sbp_24"  # Assign the string manually because numbers are discarded by `auto`
    """Systolic Blood Pressure (SBP) over 24 hours."""
    dbp_24 = "dbp_24"  # Assign the string manually because numbers are discarded by `auto`
    """Diastolic Blood Pressure (SBP) over 24 hours."""
    pp_24 = "pp_24"  # Assign the string manually because numbers are discarded by `auto`
    """Pulse Pressure (PP) over 24 hours."""
    sbp_day = auto()
    """Systolic Blood Pressure (SBP) during the day."""
    dbp_day = auto()
    """Diastolic Blood Pressure (SBP) during the day."""
    pp_day = auto()
    """Pulse Pressure (PP) during the day."""
    sbp_night = auto()
    """Systolic Blood Pressure (SBP) during the night."""
    dbp_night = auto()
    """Diastolic Blood Pressure (SBP) during the night."""
    pp_night = auto()
    """Pulse Pressure (PP) during the night."""
    ht_severity = auto()
    """How noticeable the hypertension was in 24h measurements, e.g. White coat HyperTension (WHT), controlled, etc."""
    ht_grade = auto()
    """Grade of HyperTension (HT) based on the 24h measurements of blood pressure."""
    sbp_tte = auto()
    """Systolic Blood Pressure (SBP) at the time of the TransThoracic Echocardiogram (TTE)."""
    dbp_tte = auto()
    """Diastolic Blood Pressure (SBP) at the time of the TransThoracic Echocardiogram (TTE)."""
    pp_tte = auto()
    """Pulse Pressure (PP) at the time of the TransThoracic Echocardiogram (TTE)."""
    hr_tte = auto()
    """Heart Rate (HR) at the time of the TransThoracic Echocardiogram (TTE)."""
    creat = auto()
    """Serum creatinine concentrations."""
    gfr = auto()
    """Glomerular Filtration Rate (GFR)."""
    nt_probnp = auto()
    """NT-proBNP."""
    nt_probnp_group = auto()
    """Categories of potential outcomes the patient belongs to w.r.t. NT-proBNP rate."""
    e_velocity = auto()
    """Peak E-wave (mitral passive inflow) velocity."""
    a_velocity = auto()
    """Peak A-wave (mitral inflow from active atrial contraction) velocity."""
    mv_dt = auto()
    """Mitral Valve (MV) Deceleration Time (DT)."""
    lateral_e_prime = auto()
    """Lateral mitral annular velocity."""
    septal_e_prime = auto()
    """Septal mitral annular velocity."""
    mean_e_prime = auto()
    """Average of the lateral and septal mitral annular velocity."""
    reduced_e_prime = auto()
    """Whether the mitral annular velocity is reduced, i.e. lateral E' < 10 or septal E' < 7."""
    e_e_prime_ratio = auto()
    """Ratio of E over e'."""
    d_dysfunction_e_e_prime_ratio = auto()
    """Whether the ratio E over e' is above 14, indicating Diastolic (D) dysfunction."""
    la_volume = auto()
    """Left Atrium (LA) volume."""
    dilated_la = auto()
    """Whether the Left Atrium (LA) is dilated compared to reference value."""
    la_area = auto()
    """Left Atrium (LA) area."""
    vmax_tr = auto()
    """Peak Tricuspid Regurgitation (TR) velocity."""
    ph_vmax_tr = auto()
    """Whether the Peak Tricuspid Regurgitation (TR) velocity is above 2.8, indicating Pulmonary Hypertension (PH)."""
    lvm_ind = auto()
    """Left Ventricular Mass (LVM) index."""
    lvh = auto()
    """Whether the patient suffers from Left Ventricular Hypertrophy (LVH)."""
    ivs_d = auto()
    """InterVentricular Septum (IVS) thickness at end-diastole (D)."""
    lvid_d = auto()
    """Left Ventricular Internal Diameter (LVID) at end-diastole (D)."""
    pw_d = auto()
    """Left ventricular Posterior Wall (PW) thickness at end-diastole (D)."""
    tapse = auto()
    """Tricuspid Annular Plane Systolic Excursion (TAPSE)."""
    s_prime = auto()
    """Peak tricuspid annulus (S') systolic velocity."""

    diastolic_dysfunction_param_sum = auto()
    """Sum of parameters indicating diastolic dysfunction,
    i.e. d_dysfunction_e_e_prime_ratio + reduced_e_prime + ph_vmax_tr + lvh."""
    diastolic_dysfunction = auto()
    """Categories of diastolic dysfunction."""
    ht_cm = auto()
    """Categories of HyperTensive CardioMyopathy (HT-CM)."""

    @classmethod
    def image_attrs(cls) -> List["TabularAttribute"]:
        """Lists the tabular attributes that are computed from the images."""
        return [
            TabularAttribute.ef,
            TabularAttribute.edv,
            TabularAttribute.esv,
            TabularAttribute.a4c_ed_sc_min,
            TabularAttribute.a4c_ed_sc_max,
            TabularAttribute.a4c_ed_lc_min,
            TabularAttribute.a4c_ed_lc_max,
            TabularAttribute.a2c_ed_ic_min,
            TabularAttribute.a2c_ed_ic_max,
            TabularAttribute.a2c_ed_ac_min,
            TabularAttribute.a2c_ed_ac_max,
        ]

    @classmethod
    def records_attrs(cls) -> List["TabularAttribute"]:
        """Lists the tabular attributes that come from the patient records."""
        return [attr for attr in cls if attr not in TabularAttribute.image_attrs()]

    @classmethod
    def categorical_attrs(cls) -> List["TabularAttribute"]:
        """Lists the tabular attributes that are categorical."""
        from vital.data.cardinal.utils.attributes import TABULAR_CAT_ATTR_LABELS

        return [attr for attr in cls if attr in TABULAR_CAT_ATTR_LABELS]

    @classmethod
    def ordinal_attrs(cls) -> List["TabularAttribute"]:
        """Lists the subset of categorical attributes that are ordinal (i.e. the ordering of classes is meaningful)."""
        return [
            TabularAttribute.tobacco,
            TabularAttribute.etiology,
            TabularAttribute.ht_severity,
            TabularAttribute.ht_grade,
            TabularAttribute.nt_probnp_group,
            TabularAttribute.diastolic_dysfunction_param_sum,
            TabularAttribute.diastolic_dysfunction,
            TabularAttribute.ht_cm,
        ]

    @classmethod
    def binary_attrs(cls) -> List["TabularAttribute"]:
        """Lists the subset of categorical attributes that only have 2 classes (e.g. bool)."""
        from vital.data.cardinal.utils.attributes import TABULAR_CAT_ATTR_LABELS

        return [attr for attr in cls.categorical_attrs() if len(TABULAR_CAT_ATTR_LABELS[attr]) == 2]

    @classmethod
    def boolean_attrs(cls) -> List["TabularAttribute"]:
        """Lists the subset of binary attributes that are boolean."""
        from vital.data.cardinal.utils.attributes import TABULAR_CAT_ATTR_LABELS

        return [attr for attr in cls.binary_attrs() if TABULAR_CAT_ATTR_LABELS[attr] == [False, True]]

    @classmethod
    def numerical_attrs(cls) -> List["TabularAttribute"]:
        """Lists the tabular attributes that are numerical/continuous."""
        from vital.data.cardinal.utils.attributes import TABULAR_CAT_ATTR_LABELS

        return [attr for attr in cls if attr not in TABULAR_CAT_ATTR_LABELS]


class CardinalTag(SnakeCaseStrEnum):
    """Tags referring to the different type of data stored."""

    # Tags describing data modalities
    time_series_attrs = auto()
    tabular_attrs = auto()

    # Tags referring to image data
    bmode = auto()
    mask = auto()
    resized_bmode = f"{bmode}_{DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]}"
    resized_mask = f"{mask}_{DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]}"

    # Tags for prefix/suffix to add for specific
    post = auto()

    # Tags referring to data attributes
    voxelspacing = auto()
