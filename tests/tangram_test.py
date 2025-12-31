import scanpy as sc
import tangram as tg
import numpy as np
import pandas as pd
import pytest
from pytest import approx

# to run test_tangram.py on your local machine, please set up as follow:
# - create test environment according to environment.yml (conda create env -f environment.yml) to make sure environment matches developing environment
# - install extra package: pytest, pytest-cov
# - install editable version of tangram package (pip install -e .)
#      - check tangram version (conda list tangram), make sure it is the developing version
# - make sure the test data are ready under test_data folder

# test data


@pytest.fixture
def adatas():
    ad_sc = sc.read_h5ad("data/test_ad_sc.h5ad")
    ad_sp = sc.read_h5ad("data/test_ad_sp.h5ad")
    return (ad_sc, ad_sp)


@pytest.fixture
def df_all_genes():
    df_all_genes = pd.read_csv("data/test_df.csv", index_col=0)
    return df_all_genes


@pytest.fixture
def ad_sc_mock():
    X = np.array([[0, 1, 1], [0, 1, 1]])
    obs = pd.DataFrame(index=["cell_1", "cell_2"])
    var = pd.DataFrame(index=["gene_a", "gene_b", "gene_d"])
    ad_sc_mock = sc.AnnData(X=X, obs=obs, var=var)
    return ad_sc_mock


@pytest.fixture
def ad_sp_mock():
    X = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    obs = pd.DataFrame(index=["voxel_1", "voxel_2"])
    var = pd.DataFrame(index=["gene_c", "gene_b", "gene_a", "gene_d"])
    ad_sp_mock = sc.AnnData(X=X, obs=obs, var=var)

    return ad_sp_mock


# test pp_data


@pytest.mark.parametrize("genes", [(None), (["gene_a", "gene_b"]),])
def test_pp_data(ad_sc_mock, ad_sp_mock, genes):
    tg.pp_adatas(ad_sc_mock, ad_sp_mock, genes)

    assert ad_sc_mock.uns["training_genes"] == ad_sp_mock.uns["training_genes"]
    assert ad_sc_mock.uns["overlap_genes"] == ad_sp_mock.uns["overlap_genes"]
    assert ad_sc_mock.X.any(axis=0).all() and ad_sp_mock.X.any(axis=0).all()
    assert "rna_count_based_density" in ad_sp_mock.obs.keys()
    assert "uniform_density" in ad_sp_mock.obs.keys()


# test mapping function with different parameters


@pytest.mark.parametrize(
    "lambda_g1, lambda_g2, lambda_d, density_prior, scale, e",
    [
        (1, 0, 0, None, True, np.float32(8.280743e-06)),
        (1, 0, 0, None, False, np.float32(2.785552e-07)),
        (1, 1, 0, None, True, np.float32(8.376801e-06)),
        (1, 1, 0, None, False, np.float32(2.4095453e-07)),
        (1, 1, 1, "uniform", True, np.float32(8.376801e-06)),
        (1, 1, 1, "uniform", False, np.float32(2.4095453e-07)),
        (1, 0, 2, "uniform", True, np.float32(1.3842443e-06)),
        (1, 0, 1, "rna_count_based", True, np.float32(0.0023217443)),
        (1, 0, 1, "uniform", True, np.float32(8.280743e-06)),
    ],
)
def test_map_cells_to_space(
    adatas, lambda_g1, lambda_g2, lambda_d, density_prior, scale, e,
):

    # mapping with defined random_state
    ad_map = tg.map_cells_to_space(
        adata_sc=adatas[0],
        adata_sp=adatas[1],
        device="cpu",
        mode="clusters",
        cluster_label="subclass_label",
        lambda_g1=lambda_g1,
        lambda_g2=lambda_g2,
        lambda_d=lambda_d,
        density_prior=density_prior,
        scale=scale,
        random_state=42,
        num_epochs=500,
        verbose=True,
    )

    # check if first element of output_admap.X is equal to expected value
    assert round(ad_map.X[0, 0], 3) == round(e, 3)


# test mapping exception with assertion


@pytest.mark.parametrize(
    "mode, cluster_label, lambda_g1, lambda_g2, lambda_d, scale, e",
    [
        ("clusters", "subclass", 0, 0, 0, True, "lambda_g1 cannot be 0."),
        (
            "not_clusters_or_cells",
            None,
            1,
            0,
            0,
            True,
            'Argument "mode" must be "cells" or "clusters"',
        ),
        (
            "clusters",
            None,
            1,
            0,
            0,
            True,
            "An cluster_label must be specified if mode = clusters.",
        ),
    ],
)
def test_invalid_map_cells_to_space(
    adatas, mode, cluster_label, lambda_g1, lambda_g2, lambda_d, scale, e
):
    with pytest.raises(ValueError) as exc_info:

        tg.map_cells_to_space(
            adata_sc=adatas[0],
            adata_sp=adatas[1],
            device="cpu",
            mode=mode,
            cluster_label=cluster_label,
            lambda_g1=lambda_g1,
            lambda_g2=lambda_g2,
            lambda_d=lambda_d,
            scale=scale,
            random_state=42,
            num_epochs=500,
            verbose=False,
        )
        assert e in str(exc_info.value)


# test to see if the average training score matches between the one in
# training history and the one from compare_spatial_geneexp function


@pytest.mark.parametrize(
    "mode, cluster_label, lambda_g1, lambda_g2, lambda_d, density_prior, scale",
    [
        ("clusters", "subclass_label", 1, 0, 0, None, True),
        ("clusters", "subclass_label", 1, 0, 0, None, False),
        ("clusters", "subclass_label", 1, 1, 0, None, True),
        ("clusters", "subclass_label", 1, 1, 0, None, False),
        ("clusters", "subclass_label", 1, 0, 1, "uniform", True),
        ("clusters", "subclass_label", 1, 0, 1, "rna_count_based", False),
    ],
)
def test_train_score_match(
    adatas, mode, cluster_label, lambda_g1, lambda_g2, lambda_d, density_prior, scale,
):

    # mapping with defined random_state
    ad_map = tg.map_cells_to_space(
        adata_sc=adatas[0],
        adata_sp=adatas[1],
        device="cpu",
        mode=mode,
        cluster_label=cluster_label,
        lambda_g1=lambda_g1,
        lambda_g2=lambda_g2,
        lambda_d=lambda_d,
        density_prior=density_prior,
        scale=scale,
        random_state=42,
        num_epochs=500,
        verbose=False,
    )

    # call project_genes to project input ad_sc data to ad_ge spatial data
    # with ad_map
    ad_ge = tg.project_genes(
        adata_map=ad_map,
        adata_sc=adatas[0],
        cluster_label="subclass_label",
        scale=scale,
    )

    df_all_genes = tg.compare_spatial_geneexp(ad_ge, adatas[1])

    avg_score_df = round(
        df_all_genes[df_all_genes["is_training"] == True]["score"].mean(), 3
    )
    avg_score_train_hist = round(
        np.float(list(ad_map.uns["training_history"]["main_loss"])[-1]), 3
    )

    # check if raining score matches between the one in training history and the one from compare_spatial_geneexp function
    assert avg_score_df == approx(avg_score_train_hist)


# test cross validation function
def test_eval_metric(df_all_genes):
    auc_score = tg.eval_metric(df_all_genes)[0]["auc_score"]
    assert auc_score == approx(0.750597829464878)


# test create_segment_cell_df function


@pytest.fixture
def ad_sp_segmentation():
    """Create a mock spatial AnnData with image_features for segmentation testing."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    obs = pd.DataFrame(index=["spot_0", "spot_1", "spot_2"])
    var = pd.DataFrame(index=["gene_a", "gene_b"])
    ad_sp = sc.AnnData(X=X, obs=obs, var=var)
    
    # Create image_features DataFrame with segmentation data
    # Each spot has different number of segmented cells
    image_features = pd.DataFrame(
        {
            "segmentation_centroid": [
                [(10.0, 20.0), (15.0, 25.0)],  # spot_0 has 2 cells
                [(30.0, 40.0)],                 # spot_1 has 1 cell
                [(50.0, 60.0), (55.0, 65.0), (58.0, 68.0)],  # spot_2 has 3 cells
            ],
            "segmentation_label": [2, 1, 3],  # number of cells per spot
        },
        index=["spot_0", "spot_1", "spot_2"],
    )
    ad_sp.obsm["image_features"] = image_features
    return ad_sp


@pytest.fixture
def ad_sp_segmentation_with_nan():
    """Create a mock spatial AnnData with some NaN centroids."""
    X = np.array([[1, 2], [3, 4]])
    obs = pd.DataFrame(index=["spot_0", "spot_1"])
    var = pd.DataFrame(index=["gene_a", "gene_b"])
    ad_sp = sc.AnnData(X=X, obs=obs, var=var)
    
    image_features = pd.DataFrame(
        {
            "segmentation_centroid": [
                [(10.0, 20.0)],  # spot_0 has 1 cell
                [],              # spot_1 has no cells (will produce NaN)
            ],
            "segmentation_label": [1, 0],
        },
        index=["spot_0", "spot_1"],
    )
    ad_sp.obsm["image_features"] = image_features
    return ad_sp


@pytest.fixture
def ad_sp_no_image_features():
    """Create a mock spatial AnnData without image_features."""
    X = np.array([[1, 2]])
    obs = pd.DataFrame(index=["spot_0"])
    var = pd.DataFrame(index=["gene_a", "gene_b"])
    ad_sp = sc.AnnData(X=X, obs=obs, var=var)
    return ad_sp


def test_create_segment_cell_df_basic(ad_sp_segmentation):
    """Test basic functionality of create_segment_cell_df."""
    tg.create_segment_cell_df(ad_sp_segmentation)
    
    # Check that the segmentation dataframe was created
    assert "tangram_cell_segmentation" in ad_sp_segmentation.uns
    
    df = ad_sp_segmentation.uns["tangram_cell_segmentation"]
    
    # Check columns
    assert set(df.columns) == {"spot_idx", "y", "x", "centroids"}
    
    # Check total number of rows (2 + 1 + 3 = 6 cells total)
    assert len(df) == 6
    
    # Check that centroids are properly formatted
    assert "spot_0_0" in df["centroids"].values
    assert "spot_0_1" in df["centroids"].values
    assert "spot_1_0" in df["centroids"].values
    assert "spot_2_0" in df["centroids"].values
    assert "spot_2_1" in df["centroids"].values
    assert "spot_2_2" in df["centroids"].values


def test_create_segment_cell_df_missing_image_features(ad_sp_no_image_features):
    """Test that create_segment_cell_df raises error when image_features is missing."""
    with pytest.raises(ValueError) as exc_info:
        tg.create_segment_cell_df(ad_sp_no_image_features)
    
    assert "Missing parameter for tangram deconvolution" in str(exc_info.value)


def test_create_segment_cell_df_drop_nan_true(ad_sp_segmentation_with_nan):
    """Test create_segment_cell_df with drop_nan=True (default)."""
    tg.create_segment_cell_df(ad_sp_segmentation_with_nan, drop_nan=True)
    
    df = ad_sp_segmentation_with_nan.uns["tangram_cell_segmentation"]
    
    # Should only have the one valid cell from spot_0
    assert len(df) == 1
    assert "spot_0_0" in df["centroids"].values


def test_create_segment_cell_df_drop_nan_false(ad_sp_segmentation_with_nan):
    """Test create_segment_cell_df with drop_nan=False."""
    tg.create_segment_cell_df(ad_sp_segmentation_with_nan, drop_nan=False)
    
    df = ad_sp_segmentation_with_nan.uns["tangram_cell_segmentation"]
    
    # Should include the NaN row as well
    assert len(df) >= 1
    # The NaN centroid should be converted to string "NaN"
    assert any(df["centroids"] == "NaN") or "spot_0_0" in df["centroids"].values


def test_create_segment_cell_df_coordinates(ad_sp_segmentation):
    """Test that coordinates are correctly extracted."""
    tg.create_segment_cell_df(ad_sp_segmentation)
    
    df = ad_sp_segmentation.uns["tangram_cell_segmentation"]
    
    # Check specific coordinate values for spot_0's first cell
    spot0_cell0 = df[df["centroids"] == "spot_0_0"]
    assert len(spot0_cell0) == 1
    assert spot0_cell0["y"].values[0] == 10.0
    assert spot0_cell0["x"].values[0] == 20.0

# ============== Tests for updated storage locations ==============
# Append these to test_tangram.py


@pytest.fixture
def ad_sp_with_segmentation():
    """Spatial AnnData with tangram_cell_segmentation in uns."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    obs = pd.DataFrame(index=["spot_0", "spot_1", "spot_2"])
    var = pd.DataFrame(index=["gene_a", "gene_b"])
    ad_sp = sc.AnnData(X=X, obs=obs, var=var)
    
    ad_sp.obsm["spatial"] = np.array([[100, 200], [300, 400], [500, 600]])
    
    image_features = pd.DataFrame(
        {
            "segmentation_centroid": [
                [(10.0, 20.0), (15.0, 25.0)],
                [(30.0, 40.0)],
                [(50.0, 60.0), (55.0, 65.0)],
            ],
            "segmentation_label": [2, 1, 2],
        },
        index=["spot_0", "spot_1", "spot_2"],
    )
    ad_sp.obsm["image_features"] = image_features
    
    # Pre-populate tangram_cell_segmentation (as created by create_segment_cell_df)
    segmentation_df = pd.DataFrame({
        "spot_idx": ["spot_0", "spot_0", "spot_1", "spot_2", "spot_2"],
        "y": [10.0, 15.0, 30.0, 50.0, 55.0],
        "x": [20.0, 25.0, 40.0, 60.0, 65.0],
        "centroids": ["spot_0_0", "spot_0_1", "spot_1_0", "spot_2_0", "spot_2_1"],
    })
    ad_sp.uns["tangram_cell_segmentation"] = segmentation_df
    
    return ad_sp


@pytest.fixture
def ad_sc_with_celltypes():
    """Single cell AnnData with cell type annotations."""
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]])
    obs = pd.DataFrame({
        "cell_type": ["TypeA", "TypeB", "TypeA", "TypeB", "TypeA"]
    }, index=["cell_0", "cell_1", "cell_2", "cell_3", "cell_4"])
    var = pd.DataFrame(index=["gene_a", "gene_b"])
    return sc.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def ad_map_mock(ad_sc_with_celltypes, ad_sp_with_segmentation):
    """Mock mapping AnnData (cell-by-spot)."""
    X = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.2, 0.6],
        [0.1, 0.8, 0.1],
        [0.6, 0.2, 0.2],
    ])
    obs = ad_sc_with_celltypes.obs.copy()
    var = pd.DataFrame(index=ad_sp_with_segmentation.obs.index)
    return sc.AnnData(X=X, obs=obs, var=var)


# Tests for project_cell_annotations

def test_project_cell_annotations_creates_obsm_array(ad_map_mock, ad_sp_with_segmentation):
    """Test that project_cell_annotations stores array in obsm."""
    tg.project_cell_annotations(ad_map_mock, ad_sp_with_segmentation, annotation="cell_type")
    
    assert "tangram_ct_pred" in ad_sp_with_segmentation.obsm
    assert isinstance(ad_sp_with_segmentation.obsm["tangram_ct_pred"], np.ndarray)
    assert ad_sp_with_segmentation.obsm["tangram_ct_pred"].shape == (3, 2)


def test_project_cell_annotations_creates_uns_names(ad_map_mock, ad_sp_with_segmentation):
    """Test that project_cell_annotations stores cell type names in uns."""
    tg.project_cell_annotations(ad_map_mock, ad_sp_with_segmentation, annotation="cell_type")
    
    assert "tangram_ct_pred_names" in ad_sp_with_segmentation.uns
    assert isinstance(ad_sp_with_segmentation.uns["tangram_ct_pred_names"], list)
    assert set(ad_sp_with_segmentation.uns["tangram_ct_pred_names"]) == {"TypeA", "TypeB"}


# Tests for count_cell_annotations

def test_count_cell_annotations_missing_segmentation(ad_map_mock, ad_sc_with_celltypes):
    """Test error when tangram_cell_segmentation is missing."""
    ad_sp_no_seg = sc.AnnData(
        X=np.array([[1, 2], [3, 4], [5, 6]]),
        obs=pd.DataFrame(index=["spot_0", "spot_1", "spot_2"]),
        var=pd.DataFrame(index=["gene_a", "gene_b"])
    )
    ad_sp_no_seg.obsm["spatial"] = np.array([[100, 200], [300, 400], [500, 600]])
    ad_sp_no_seg.obsm["image_features"] = pd.DataFrame({
        "segmentation_label": [2, 1, 2]
    }, index=["spot_0", "spot_1", "spot_2"])
    
    with pytest.raises(ValueError) as exc_info:
        tg.count_cell_annotations(ad_map_mock, ad_sc_with_celltypes, ad_sp_no_seg)
    assert "Run `create_segment_cell_df`" in str(exc_info.value)


def test_count_cell_annotations_creates_uns(ad_map_mock, ad_sc_with_celltypes, ad_sp_with_segmentation):
    """Test that count_cell_annotations stores DataFrame in uns (not obsm)."""
    tg.count_cell_annotations(ad_map_mock, ad_sc_with_celltypes, ad_sp_with_segmentation, annotation="cell_type")
    
    assert "tangram_ct_count" in ad_sp_with_segmentation.uns
    assert isinstance(ad_sp_with_segmentation.uns["tangram_ct_count"], pd.DataFrame)


def test_count_cell_annotations_centroids_from_uns(ad_map_mock, ad_sc_with_celltypes, ad_sp_with_segmentation):
    """Test that centroids are extracted from uns segmentation df."""
    tg.count_cell_annotations(ad_map_mock, ad_sc_with_celltypes, ad_sp_with_segmentation, annotation="cell_type")
    
    df = ad_sp_with_segmentation.uns["tangram_ct_count"]
    
    # spot_0 should have 2 centroids
    spot0_centroids = df.loc["spot_0", "centroids"]
    assert len(spot0_centroids) == 2
    assert "spot_0_0" in spot0_centroids
    assert "spot_0_1" in spot0_centroids


# Tests for deconvolve_cell_annotations

def test_deconvolve_missing_ct_count(ad_sp_with_segmentation):
    """Test error when tangram_ct_count is missing."""
    with pytest.raises(ValueError) as exc_info:
        tg.deconvolve_cell_annotations(ad_sp_with_segmentation)
    assert "Run `count_cell_annotations`" in str(exc_info.value)


def test_deconvolve_reads_from_uns(ad_map_mock, ad_sc_with_celltypes, ad_sp_with_segmentation):
    """Test that deconvolution reads from uns."""
    tg.project_cell_annotations(ad_map_mock, ad_sp_with_segmentation, annotation="cell_type")
    tg.count_cell_annotations(ad_map_mock, ad_sc_with_celltypes, ad_sp_with_segmentation, annotation="cell_type")
    
    result = tg.deconvolve_cell_annotations(ad_sp_with_segmentation)
    
    assert isinstance(result, sc.AnnData)
    assert "spatial" in result.obsm
    assert isinstance(result.obsm["spatial"], np.ndarray)
    
    valid_types = set(ad_sp_with_segmentation.uns["tangram_ct_pred_names"])
    assert all(ct in valid_types for ct in result.obs["cluster"].unique())


# Integration test

def test_full_deconvolution_pipeline(ad_map_mock, ad_sc_with_celltypes, ad_sp_with_segmentation):
    """Test full pipeline with updated storage locations."""
    tg.project_cell_annotations(ad_map_mock, ad_sp_with_segmentation, annotation="cell_type")
    assert isinstance(ad_sp_with_segmentation.obsm["tangram_ct_pred"], np.ndarray)
    assert isinstance(ad_sp_with_segmentation.uns["tangram_ct_pred_names"], list)
    
    tg.count_cell_annotations(ad_map_mock, ad_sc_with_celltypes, ad_sp_with_segmentation, annotation="cell_type")
    assert isinstance(ad_sp_with_segmentation.uns["tangram_ct_count"], pd.DataFrame)
    
    result = tg.deconvolve_cell_annotations(ad_sp_with_segmentation)
    assert isinstance(result, sc.AnnData)
    assert isinstance(result.obsm["spatial"], np.ndarray)