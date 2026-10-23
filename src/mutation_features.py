"""Load METABRIC and turn its *_mut columns into numeric feature matrices.

Each *_mut cell is "0" (no mutation called) or one or more space-separated
protein-change codes (e.g. "H1047R", "R306*", "P322Hfs*23", "X120_splice").

    * load_metabric - read the METABRIC csv and check required columns
    * er_labels - 0/1 ER status vector (1 = ER+)
    * mutations_long - one row per (patient, gene, single mutation)
    * classify_mutation - map a protein-change code to a consequence class
    * indicator_matrix - patients x keys 0/1 matrix
    * build_feature_sets - the encodings compared in the analysis
"""

import re

import pandas as pd

# Order matters: first matching pattern wins.
MUTATION_CLASSES = {
    "truncating": r"(\*|fs)",            # nonsense (R306*) or frameshift (fs)
    "splice": r"splice",                 # X120_splice
    "inframe": r"(?:del|ins|dup)",       # I33del, L127_L133del
    "silent": r"=$",                     # D947= (synonymous)
    "missense": r"^[A-Z]\d+[A-Z]$",      # H1047R
}

LABEL_COLUMN = "er_status"
MUT_SUFFIX = "_mut"


def load_metabric(data_file):
    """Read the METABRIC csv and check that the needed columns exist.

    Parameters
    ----------
    data_file : str or Path
        Path to METABRIC_RNA_Mutation.csv.

    Returns
    -------
    df : pandas.DataFrame
        One row per patient.

    Raises
    ------
    FileNotFoundError
        If `data_file` does not exist.
    KeyError
        If the label column or the *_mut columns are missing.
    """
    df = pd.read_csv(data_file, low_memory=False)
    if LABEL_COLUMN not in df.columns:
        raise KeyError(f"column '{LABEL_COLUMN}' not found in {data_file}")
    if not df.columns.str.endswith(MUT_SUFFIX).any():
        raise KeyError(f"no '*{MUT_SUFFIX}' columns found in {data_file}")
    return df


def er_labels(df):
    """Return ER status as a 0/1 Series (1 = ER positive)."""
    return (df[LABEL_COLUMN] == "Positive").astype(int)


def mutations_long(df):
    """Reshape the *_mut columns to one row per (patient, gene, mutation).

    Parameters
    ----------
    df : pandas.DataFrame
        METABRIC table from `load_metabric`.

    Returns
    -------
    long : pandas.DataFrame
        Columns 'patient' (row index of `df`), 'gene', 'mutation' and
        'mclass' (consequence class from `classify_mutation`).
    """
    mut_cols = df.columns[df.columns.str.endswith(MUT_SUFFIX)]
    long = (
        df[mut_cols].astype(str)
        .stack()
        .rename_axis(["patient", "gene"])
        .rename("mutation")
        .reset_index()
    )
    long = long[long["mutation"] != "0"]
    long["mutation"] = long["mutation"].str.split()
    long = long.explode("mutation").reset_index(drop=True)
    long["gene"] = long["gene"].str.removesuffix(MUT_SUFFIX)
    long["mclass"] = long["mutation"].map(classify_mutation)
    return long


def classify_mutation(code):
    """Map a protein-change code (e.g. 'R306*') to a consequence class.

    Parameters
    ----------
    code : str
        A single protein-change code.

    Returns
    -------
    mclass : str
        A key of MUTATION_CLASSES, or 'other' if no pattern matches.
    """
    for name, pattern in MUTATION_CLASSES.items():
        if re.search(pattern, code):
            return name
    return "other"


def indicator_matrix(long, keys, patients, min_count):
    """Build a patients x keys 0/1 matrix.

    Parameters
    ----------
    long : pandas.DataFrame
        Output of `mutations_long`.
    keys : pandas.Series
        Feature name for each row of `long` (e.g. gene, or gene_class).
    patients : pandas.Index
        All patients; those with no mutations get a row of zeros.
    min_count : int
        Columns observed in fewer than `min_count` patients are dropped.

    Returns
    -------
    mat : pandas.DataFrame
        0/1 indicator matrix indexed by `patients`.
    """
    mat = pd.crosstab(long["patient"], keys).clip(upper=1)
    mat = mat.reindex(patients, fill_value=0)
    return mat.loc[:, mat.sum() >= min_count]


def build_feature_sets(df, min_gene_freq, min_hotspot):
    """Build the mutation encodings compared in the analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        METABRIC table from `load_metabric`.
    min_gene_freq : int
        Minimum number of patients for a gene / gene-class feature.
    min_hotspot : int
        Minimum number of patients for a recurrent-allele feature.

    Returns
    -------
    feature_sets : dict of str to pandas.DataFrame
        'TP53 only' (baseline), 'Gene', 'Gene x class' and
        'Gene x class + hotspot'.
    """
    long = mutations_long(df)
    gene = indicator_matrix(long, long["gene"], df.index, min_gene_freq)
    gene_class = indicator_matrix(
        long, long["gene"] + "_" + long["mclass"], df.index, min_gene_freq)
    hotspot = indicator_matrix(
        long, long["gene"] + "_" + long["mutation"], df.index, min_hotspot)
    return {
        "TP53 only": gene[["tp53"]],
        "Gene": gene,
        "Gene x class": gene_class,
        "Gene x class + hotspot": pd.concat([gene_class, hotspot], axis=1),
    }
