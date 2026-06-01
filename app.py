# ============================================================
# STREAMLIT GUI: POINT-WISE GP-CDI SCENARIO PREDICTION
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pickle
import folium
from streamlit_folium import st_folium
from folium.plugins import MeasureControl, MousePosition, Fullscreen, MiniMap


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="GP-CDI Scenario Tool",
    layout="wide"
)

st.title("Interactive Intensity Scenario Prediction Explorer")

st.markdown(
    """
    This interactive graphical user interface (GUI) predicts **Community Decimal Intensity (CDI)** 
    for user-defined earthquake scenarios using a trained Bayesian Gaussian Process (GP) model 
    developed for the Bengal Basin region.

    The framework performs probabilistic CDI prediction, posterior uncertainty quantification, 
    exceedance probability estimation, and interactive spatial visualization for scenario-based 
    seismic hazard assessment in data-sparse regions.

    ---
    ### Cite

    Shams, R., & Mohanty, W. K. (20XX).  
    A Framework for Using Did You Feel it? (DYFI) Community Decimal Intensity in Data-Sparse Urban Seismic Hazard Assessment for Eastern India (Bengal Basin).  
    (Submitted to Bulletin of Earthquake Engineering).
    """
)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    idata_loaded = az.from_netcdf("idata_gp_spatial_fast_allDataCDI.nc")

    with open("model_gp_spatial_fast_meta_allDataCDI.pkl", "rb") as f:
        model_meta = pickle.load(f)

    return idata_loaded, model_meta


@st.cache_data
def load_training_data():
    return pd.read_csv("df_gp.csv")


idata_loaded, model_meta = load_model()
df_gp = load_training_data()


# ------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------

def rbf_kernel(X1, X2, eta, ell, jitter=0.0):
    sqdist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
    K = (eta ** 2) * np.exp(-0.5 * sqdist / (ell ** 2))

    if jitter > 0 and X1.shape[0] == X2.shape[0]:
        K += jitter * np.eye(X1.shape[0])

    return K


def haversine_km(lon1, lat1, lon2, lat2):
    R_earth = 6371.0

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arcsin(np.sqrt(a))

    return R_earth * c


def build_scenario_point(site_lon, site_lat, source_lon, source_lat, mag, depth_km):
    epi_dist_km = haversine_km(source_lon, source_lat, site_lon, site_lat)
    hypo_dist_km = np.sqrt(epi_dist_km ** 2 + depth_km ** 2)

    df_new = pd.DataFrame({
        "Longitude": [site_lon],
        "Latitude": [site_lat],
        "Mag": [mag],
        "Depth_km": [depth_km],
        "Epicentral distance": [epi_dist_km],
        "Hypocentral distance": [hypo_dist_km],
        "No. of responses": [10]
    })

    return df_new


def predict_gp_fast_point_posterior(
    df_train,
    df_new,
    idata_loaded,
    model_meta,
    n_draws=100,
    jitter=1e-6
):
    posterior = idata_loaded.posterior
    use_depth = model_meta.get("use_depth", True)

    X_train = df_train[["Longitude", "Latitude"]].values.astype(float)
    X_train_scaled = (
        X_train - model_meta["X_space_mean"]
    ) / model_meta["X_space_std"]

    X_new = df_new[["Longitude", "Latitude"]].values.astype(float)
    X_new_scaled = (
        X_new - model_meta["X_space_mean"]
    ) / model_meta["X_space_std"]

    R_train = df_train["Hypocentral distance"].values.astype(float)
    M_train = df_train["Mag"].values.astype(float)

    R_new = df_new["Hypocentral distance"].values.astype(float)
    M_new = df_new["Mag"].values.astype(float)

    logR_train_raw = np.log(R_train + 1.0)
    logR_new_raw = np.log(R_new + 1.0)

    if "logR_mean_raw" in model_meta and "logR_std_raw" in model_meta:
        logR_new_std = (
            logR_new_raw - model_meta["logR_mean_raw"]
        ) / model_meta["logR_std_raw"]
    else:
        logR_new_std = (
            logR_new_raw - np.mean(logR_train_raw)
        ) / np.std(logR_train_raw)

    if "M_mean_raw" in model_meta and "M_std_raw" in model_meta:
        M_new_std = (
            M_new - model_meta["M_mean_raw"]
        ) / model_meta["M_std_raw"]
    else:
        M_new_std = (
            M_new - np.mean(M_train)
        ) / np.std(M_train)

    if use_depth:
        depth_new = df_new["Depth_km"].values.astype(float)

        if "depth_mean_raw" in model_meta and "depth_std_raw" in model_meta:
            depth_new_std = (
                depth_new - model_meta["depth_mean_raw"]
            ) / model_meta["depth_std_raw"]
        else:
            depth_train = df_train["Depth_km"].values.astype(float)
            depth_new_std = (
                depth_new - np.mean(depth_train)
            ) / np.std(depth_train)

    alpha_all = posterior["alpha"].stack(sample=("chain", "draw")).values
    beta_logR_all = posterior["beta_logR"].stack(sample=("chain", "draw")).values
    beta_M_all = posterior["beta_M"].stack(sample=("chain", "draw")).values
    eta_all = posterior["eta_gp"].stack(sample=("chain", "draw")).values
    ell_all = posterior["ell_gp"].stack(sample=("chain", "draw")).values
    f_space_all = posterior["f_space"].stack(sample=("chain", "draw")).values

    if f_space_all.shape[0] != X_train.shape[0] and f_space_all.shape[1] == X_train.shape[0]:
        f_space_all = f_space_all.T

    if use_depth:
        if "beta_depth" not in posterior:
            raise ValueError("model_meta indicates use_depth=True, but beta_depth not found in posterior.")
        beta_depth_all = posterior["beta_depth"].stack(sample=("chain", "draw")).values

    n_samples_total = alpha_all.shape[0]
    n_draws_use = min(n_draws, n_samples_total)

    idx = np.linspace(0, n_samples_total - 1, n_draws_use, dtype=int)

    preds = np.zeros(n_draws_use)

    for k, i in enumerate(idx):
        mu_fixed = (
            alpha_all[i]
            + beta_logR_all[i] * logR_new_std
            + beta_M_all[i] * M_new_std
        )

        if use_depth:
            mu_fixed += beta_depth_all[i] * depth_new_std

        K_xx = rbf_kernel(
            X_train_scaled,
            X_train_scaled,
            eta_all[i],
            ell_all[i],
            jitter=jitter
        )

        K_xs = rbf_kernel(
            X_train_scaled,
            X_new_scaled,
            eta_all[i],
            ell_all[i]
        )

        a = np.linalg.solve(K_xx, f_space_all[:, i])
        f_star = K_xs.T @ a

        preds[k] = mu_fixed[0] + f_star[0]

    return preds



# ------------------------------------------------------------
# RECENT EARTHQUAKE SCENARIOS
# ------------------------------------------------------------

recent_events = [
    {
        "event_name": "2025 Mw 5.4 Tungi, Bangladesh earthquake",
        "time": "2025-11-21T04:38:28.942Z",
        "latitude": 23.8580,
        "longitude": 90.5404,
        "depth_km": 27.0,
        "magnitude": 5.4,
        "mag_type": "Mw",
        "location": "14 km ESE of Tungi, Bangladesh"
    },
    {
        "event_name": "2026 M 5.3 Taki, India earthquake",
        "time": "2026-02-27T07:52:24.828Z",
        "latitude": 22.4510,
        "longitude": 89.1394,
        "depth_km": 9.751,
        "magnitude": 5.3,
        "mag_type": "mb",
        "location": "26 km SE of Taki, India"
    }
]

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

if "run_prediction" not in st.session_state:
    st.session_state.run_prediction = False

if "results" not in st.session_state:
    st.session_state.results = None


# ------------------------------------------------------------
# SIDEBAR GUI INPUTS
# ------------------------------------------------------------

st.sidebar.header("User Inputs")

# ------------------------------------------------------------
# SCENARIO PRESETS
# ------------------------------------------------------------

st.sidebar.subheader("Scenario Preset")

event_options = ["Custom scenario"] + [
    event["event_name"] for event in recent_events
]

selected_event_name = st.sidebar.selectbox(
    "Select recent earthquake scenario",
    event_options
)

selected_event = None

if selected_event_name != "Custom scenario":
    selected_event = next(
        event for event in recent_events
        if event["event_name"] == selected_event_name
    )

    st.sidebar.info(
        f"""
        **{selected_event['event_name']}**  
        Time: {selected_event['time']}  
        Location: {selected_event['location']}  
        Magnitude: {selected_event['magnitude']} {selected_event['mag_type']}  
        Depth: {selected_event['depth_km']} km
        """
    )

default_source_lat = selected_event["latitude"] if selected_event else 23.5000
default_source_lon = selected_event["longitude"] if selected_event else 90.0000
default_mag = selected_event["magnitude"] if selected_event else 6.0
default_depth_km = selected_event["depth_km"] if selected_event else 30.0

# ------------------------------------------------------------
# SITE INPUTS
# ------------------------------------------------------------

st.sidebar.subheader("Prediction Site")

site_lat = st.sidebar.number_input(
    "Site latitude",
    value=22.5726,
    format="%.4f"
)

site_lon = st.sidebar.number_input(
    "Site longitude",
    value=88.3639,
    format="%.4f"
)

# ------------------------------------------------------------
# SOURCE INPUTS
# ------------------------------------------------------------

st.sidebar.subheader("Earthquake Source")

source_lat = st.sidebar.number_input(
    "Source latitude",
    value=float(default_source_lat),
    format="%.4f"
)

source_lon = st.sidebar.number_input(
    "Source longitude",
    value=float(default_source_lon),
    format="%.4f"
)

mag = st.sidebar.number_input(
    "Magnitude Mw",
    min_value=3.0,
    max_value=9.5,
    value=float(default_mag),
    step=0.1
)

depth_km = st.sidebar.number_input(
    "Focal depth, km",
    min_value=0.0,
    max_value=300.0,
    value=float(default_depth_km),
    step=1.0
)

n_draws = st.sidebar.slider(
    "Posterior draws",
    min_value=20,
    max_value=500,
    value=100,
    step=20
)

run_button = st.sidebar.button("Run GP-CDI Prediction")

if run_button:
    st.session_state.run_prediction = True

if st.sidebar.button("Reset Results"):
    st.session_state.run_prediction = False
    st.session_state.results = None
    st.rerun()


# ------------------------------------------------------------
# DEFAULT INFO PANEL
# ------------------------------------------------------------

if not st.session_state.run_prediction and st.session_state.results is None:
    st.info("Enter the site and earthquake source parameters in the sidebar, then click **Run GP-CDI Prediction**.")

    st.markdown(
        """
        **Example workflow**

        - Select **Custom scenario** or one of the recent earthquake presets.  
        - Default prediction site: Kolkata  
        - Site latitude: 22.5726  
        - Site longitude: 88.3639  
        - Available presets: 2025 Tungi, Bangladesh Mw 5.4 and 2026 Taki, India M 5.3.  
        """
    )


# ------------------------------------------------------------
# RUN MODEL AND STORE RESULTS
# ------------------------------------------------------------

if st.session_state.run_prediction and run_button:

    df_point = build_scenario_point(
        site_lon=site_lon,
        site_lat=site_lat,
        source_lon=source_lon,
        source_lat=source_lat,
        mag=mag,
        depth_km=depth_km
    )

    with st.spinner("Running posterior GP-CDI prediction..."):
        cdi_post = predict_gp_fast_point_posterior(
            df_train=df_gp,
            df_new=df_point,
            idata_loaded=idata_loaded,
            model_meta=model_meta,
            n_draws=n_draws
        )

    st.session_state.results = {
        "df_point": df_point,
        "cdi_post": cdi_post,
        "site_lat": site_lat,
        "site_lon": site_lon,
        "source_lat": source_lat,
        "source_lon": source_lon,
        "mag": mag,
        "depth_km": depth_km,
        "n_draws": n_draws,
        "selected_event_name": selected_event_name
    }


# ------------------------------------------------------------
# DISPLAY STORED RESULTS
# ------------------------------------------------------------

if st.session_state.results is not None:

    results = st.session_state.results

    df_point = results["df_point"]
    cdi_post = results["cdi_post"]

    site_lat = results["site_lat"]
    site_lon = results["site_lon"]
    source_lat = results["source_lat"]
    source_lon = results["source_lon"]
    mag = results["mag"]
    depth_km = results["depth_km"]
    n_draws = results["n_draws"]
    selected_event_name = results.get("selected_event_name", "Custom scenario")

    mean_cdi = np.mean(cdi_post)
    median_cdi = np.median(cdi_post)
    std_cdi = np.std(cdi_post, ddof=1)
    lower95 = np.percentile(cdi_post, 2.5)
    upper95 = np.percentile(cdi_post, 97.5)

    epi_dist = df_point["Epicentral distance"].iloc[0]
    hypo_dist = df_point["Hypocentral distance"].iloc[0]

    st.subheader("Prediction Summary")

    if selected_event_name != "Custom scenario":
        st.markdown(f"**Selected scenario preset:** {selected_event_name}")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Mean CDI", f"{mean_cdi:.2f}")
    col2.metric("Median CDI", f"{median_cdi:.2f}")
    col3.metric("Std CDI", f"{std_cdi:.2f}")
    col4.metric("Epicentral Distance", f"{epi_dist:.1f} km")
    col5.metric("Hypocentral Distance", f"{hypo_dist:.1f} km")

    st.markdown(f"**95% credible interval:** [{lower95:.2f}, {upper95:.2f}]")

    thresholds = [2, 3, 4, 5, 6]

    exc_df = pd.DataFrame({
        "CDI threshold": thresholds,
        "Exceedance probability": [np.mean(cdi_post > c) for c in thresholds]
    })

    st.subheader("CDI Exceedance Probabilities")
    st.dataframe(exc_df, use_container_width=True)

    tab1, tab2, tab3 = st.tabs([
        "Interactive Map",
        "Posterior Analysis",
        "Download Results"
    ])

    # --------------------------------------------------------
    # TAB 1: LIGHT MAP
    # --------------------------------------------------------

    with tab1:
        st.subheader("Interactive Scenario Map")

        center_lat = (site_lat + source_lat + 22.5726) / 3
        center_lon = (site_lon + source_lon + 88.3639) / 3

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles="CartoDB positron",
            control_scale=True
        )

        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
        folium.TileLayer("CartoDB positron", name="Light Map").add_to(m)

        folium.Circle(
            location=[22.5726, 88.3639],
            radius=300 * 1000,
            color="purple",
            fill=False,
            weight=3,
            tooltip="300 km radius around Kolkata"
        ).add_to(m)

        folium.Marker(
            location=[22.5726, 88.3639],
            popup="<b>Kolkata</b><br>Reference center for 300 km study region",
            tooltip="Kolkata",
            icon=folium.Icon(color="purple", icon="home")
        ).add_to(m)

        for r in [50, 100, 200, 300]:
            folium.Circle(
                location=[source_lat, source_lon],
                radius=r * 1000,
                color="gray",
                fill=False,
                weight=1,
                opacity=0.45,
                tooltip=f"{r} km from source"
            ).add_to(m)

        if mean_cdi < 2:
            site_color = "green"
        elif mean_cdi < 4:
            site_color = "orange"
        else:
            site_color = "red"

        site_popup = f"""
        <b>Prediction Site</b><br>
        Latitude: {site_lat:.4f}<br>
        Longitude: {site_lon:.4f}<br><br>
        <b>CDI Prediction</b><br>
        Mean CDI: {mean_cdi:.2f}<br>
        Median CDI: {median_cdi:.2f}<br>
        Std CDI: {std_cdi:.2f}<br>
        95% CI: [{lower95:.2f}, {upper95:.2f}]<br><br>
        <b>Exceedance</b><br>
        P(CDI &gt; 2): {np.mean(cdi_post > 2):.3f}<br>
        P(CDI &gt; 3): {np.mean(cdi_post > 3):.3f}<br>
        P(CDI &gt; 4): {np.mean(cdi_post > 4):.3f}<br>
        P(CDI &gt; 5): {np.mean(cdi_post > 5):.3f}
        """

        folium.Marker(
            location=[site_lat, site_lon],
            popup=folium.Popup(site_popup, max_width=350),
            tooltip=f"Prediction Site | Mean CDI = {mean_cdi:.2f}",
            icon=folium.Icon(color=site_color, icon="info-sign")
        ).add_to(m)

        source_popup = f"""
        <b>Earthquake Source</b><br>
        Latitude: {source_lat:.4f}<br>
        Longitude: {source_lon:.4f}<br>
        Magnitude: Mw {mag:.2f}<br>
        Depth: {depth_km:.1f} km<br><br>
        Epicentral distance to site: {epi_dist:.1f} km<br>
        Hypocentral distance to site: {hypo_dist:.1f} km
        """

        folium.Marker(
            location=[source_lat, source_lon],
            popup=folium.Popup(source_popup, max_width=350),
            tooltip=f"Earthquake Source | Mw {mag:.2f}",
            icon=folium.Icon(color="red", icon="star")
        ).add_to(m)

        folium.PolyLine(
            locations=[
                [source_lat, source_lon],
                [site_lat, site_lon]
            ],
            color="black",
            weight=3,
            opacity=0.7,
            tooltip=f"Epicentral distance = {epi_dist:.1f} km"
        ).add_to(m)

        m.add_child(MeasureControl())
        m.add_child(Fullscreen())
        m.add_child(MiniMap())

        MousePosition(
            position="bottomright",
            separator=" | ",
            prefix="Coordinates:"
        ).add_to(m)

        folium.LayerControl().add_to(m)

        st_folium(m, width=1200, height=650)

    # --------------------------------------------------------
    # TAB 2: POSTERIOR ANALYSIS
    # --------------------------------------------------------

    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Posterior Predictive Distribution")

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(cdi_post, bins=25, density=True, alpha=0.7, edgecolor="black")
            ax.axvline(mean_cdi, linestyle="--", linewidth=2, label=f"Mean = {mean_cdi:.2f}")
            ax.axvline(lower95, linestyle=":", linewidth=2, label="95% CI")
            ax.axvline(upper95, linestyle=":", linewidth=2)
            ax.axvspan(lower95, upper95, alpha=0.15)
            ax.set_xlabel("CDI")
            ax.set_ylabel("Posterior density")
            ax.set_title("Posterior CDI Distribution")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        with col_b:
            st.subheader("CDI Exceedance Curve")

            cdi_values = np.linspace(
                np.floor(np.min(cdi_post)),
                np.ceil(np.max(cdi_post)),
                100
            )

            exceed_probs = [np.mean(cdi_post > c) for c in cdi_values]

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.plot(cdi_values, exceed_probs, linewidth=2)
            ax2.set_xlabel("CDI threshold")
            ax2.set_ylabel("P(CDI > c)")
            ax2.set_title("CDI Exceedance Curve")
            ax2.set_ylim(0, 1.05)
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)

    # --------------------------------------------------------
    # TAB 3: DOWNLOAD RESULTS
    # --------------------------------------------------------

    with tab3:
        st.subheader("Download Prediction Results")

        result_df = df_point.copy()
        result_df["scenario_preset"] = selected_event_name
        result_df["CDI_mean"] = mean_cdi
        result_df["CDI_median"] = median_cdi
        result_df["CDI_std"] = std_cdi
        result_df["CDI_2.5%"] = lower95
        result_df["CDI_97.5%"] = upper95

        for c in thresholds:
            result_df[f"P_CDI_gt_{c}"] = np.mean(cdi_post > c)

        st.dataframe(result_df, use_container_width=True)

        st.download_button(
            label="Download CDI prediction CSV",
            data=result_df.to_csv(index=False),
            file_name="pointwise_CDI_prediction.csv",
            mime="text/csv"
        )