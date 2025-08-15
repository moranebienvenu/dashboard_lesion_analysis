import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from scipy.stats import ttest_ind, ttest_rel, levene, mannwhitneyu, shapiro, wilcoxon, norm
import io
import zipfile
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import colorsys
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Tweedie, Gaussian, Gamma, Poisson
from statsmodels.genmod.families.links import Log, Identity, InversePower, Sqrt, NegativeBinomial
from nilearn import plotting
import seaborn as sns
from statsmodels.stats.power import TTestIndPower, TTestPower
from itertools import combinations
from scipy.stats import shapiro, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from io import BytesIO

#Instruction 
#Nom des sujets doit etre sous cette forme : sub-[A-Z0-9]_ses-V[123] exemple : sub-A1503_ses-V1 ou sub-A1_ses-V1 avec A1503 le numero du patient et V1 la session


# -------- Used functions --------

def process_zip_to_dataframe(uploaded_zip):
    """Lit un ZIP NeuroTmap avec tous les fichiers csv "output_les_dis_'lesion_name'" et " output_pre_post_synaptic_ratio_'lesion_name'
    et retourne un DataFrame combiné sujet par sujet 
    Si présence d'un fichier csv ou excel nommé "clinical_data.csv/xlsx" alors concatener aussi sur le nom des sujets au fichier global"""

    # Dictionnaires de DataFrame par type
    data_les_dis = {}
    data_pre_post = {}
    df_clinical = None
    missing_clinical_subjects = []
    systems_list = []

    with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as z:
        for filename in z.namelist():
            # ⛔️ Ignorer tous les fichiers systèmes ou non pertinents
            if (
                "__MACOSX" in filename 
                or filename.endswith(".DS_Store") 
                or "/._" in filename 
                or (not "output_les_dis" in filename and not "output_pre_post_synaptic_ratio" in filename and "clinical_data" not in filename)
            ):
                continue

            if filename.endswith(".csv") or filename.endswith(".xlsx"):
                # Cas du fichier clinique
                if "clinical_data" in filename:
                    with z.open(filename) as f:
                        try:
                            if filename.endswith(".csv"):
                                content = f.read().decode("utf-8")
                                if "," in content and " " in content:
                                    df_clinical = pd.read_csv(io.StringIO(content), sep=" ", decimal=",")
                                else:
                                    df_clinical = pd.read_csv(io.StringIO(content))
                            else:
                                df_clinical = pd.read_excel(f, engine="openpyxl")
                            st.success("✅ Clinical file loaded")
                        except Exception as e:
                            st.warning(f"❌ Error loading clinical file: {e}")
                    continue  


            # Extraction du subject : sub-XXX_ses-VX
            if "output_les_dis" in filename or "output_pre_post_synaptic_ratio" in filename:
                match = re.search(r"sub-[A-Za-z0-9]+_ses-V[0-9]+", filename)
                if not match:
                    st.warning(f"❌ Subject name not recognized in: {filename}")
                    continue

            subject = match.group(0)

            with z.open(filename) as f:
                try:
                    df = pd.read_csv(f, sep=' ', index_col=0)
                except Exception as e:
                    st.warning(f"❌ Read error {filename}: {e}")
                    continue

            # Classement par type
            if "output_les_dis" in filename:
                # On attend que df ait une index colonne et des colonnes systèmes
                        # Par exemple, index = ['loc_inj_sub-A1503_ses-V2', ...]
                row = {"subject": subject}
                # Détection des systèmes (colonnes du fichier)
                if not systems_list and len(df.columns) > 0:
                    systems_list = list(df.columns)
                for idx in df.index:
                    prefix = str(idx).split('_sub-')[0]  # ex: loc_inj 
                    for system in df.columns:
                        colname = f"{prefix}_{system}"
                        row[colname] = df.at[idx, system]
                data_les_dis[subject] = pd.DataFrame([row])  

            elif "output_pre_post_synaptic_ratio" in filename:
                 # Ici, on va parcourir les colonnes, qui sont de la forme "A4B2 presynaptic", "VAChT postsynaptic", etc.
                        # On crée un dictionnaire pour la ligne du sujet
                        row = {"subject": subject}

                        for col in df.columns:
                            # Ex: "A4B2 presynaptic" -> pre_A4B2
                            # Ex: "VAChT postsynaptic" -> post_VAChT
                            m = re.match(r"(.+?)\s+(presynaptic|postsynaptic)", col)
                            if m:
                                system_name = m.group(1).strip()
                                syn_type = m.group(2).strip()
                                prefix = "pre" if syn_type == "presynaptic" else "post"
                                new_colname = f"{prefix}_{system_name}"
                                val = df[col].iloc[0] if not df.empty else None
                                row[new_colname] = val
                        data_pre_post[subject] = pd.DataFrame([row])
    
            

    # Fusion sujet par sujet
    combined_rows = []
    all_subjects = set(data_les_dis.keys()) | set(data_pre_post.keys()) 

    for subject in sorted(all_subjects):
        row = {"subject": subject}

        if subject in data_les_dis:
            try:
                row.update(data_les_dis[subject].iloc[0].to_dict())
            except Exception as e:
                st.warning(f"❌ Data extraction error for les_dis {subject}: {e}")

        # Ajouter les données pre_post
        if subject in data_pre_post:
            try:
                row.update(data_pre_post[subject].iloc[0].to_dict())
            except Exception as e:
                st.warning(f"❌ Data extraction error for pre_post {subject}: {e}")

        # Ajouter les données cliniques si disponibles
        if df_clinical is not None:
            match_row = df_clinical[df_clinical['subject'] == subject]
            if not match_row.empty:
                row.update(match_row.iloc[0].to_dict())
            else:
                missing_clinical_subjects.append(subject)

        combined_rows.append(row)
    # Création du DataFrame final
    final_df = pd.DataFrame(combined_rows)
    
    # Réorganisation des colonnes pour avoir un ordre logique
    if systems_list:
        ordered_columns = ['subject']
        
        # Ajouter les colonnes les_dis dans l'ordre: loc_inj, loc_inj_perc, tract_inj, tract_inj_perc pour chaque système
        for system in systems_list:
            for measure in ['loc_inj', 'loc_inj_perc', 'tract_inj', 'tract_inj_perc']:
                colname = f"{measure}_{system}"
                if colname in final_df.columns:
                    ordered_columns.append(colname)
        
        # Ajouter les colonnes pre/post a la suite des colonnes précédentes
        pre_post_columns = [col for col in final_df.columns if col.startswith(('pre_', 'post_')) and col not in ordered_columns]
        ordered_columns.extend(sorted(pre_post_columns))
        
        # Ajouter les colonnes cliniques restantes
        other_columns = [col for col in final_df.columns if col not in ordered_columns and col != 'subject']
        ordered_columns.extend(other_columns)
        
        final_df = final_df[ordered_columns]
        if missing_clinical_subjects:
                st.info("ℹ️ No clinical data found for the following subjects: " + ", ".join(missing_clinical_subjects))

    return final_df
    
def create_interactive_plots(df, subjects, title_suffix="", is_group=False, is_overlay=False):

    # Filtrer les données et calculer les moyennes que si groupe sinon mettre les données individuelles pour chaque sujet base/overlay
    plot_data = df[df['subject'].isin(subjects)]
    # Détection dynamique des systèmes
    systems = [col.replace('loc_inj_', '') for col in df.columns 
               if col.startswith('loc_inj_') and not col.startswith('loc_inj_perc')]
    # Colonnes d'intérêt pour la moyenne
    loc_cols = [f'loc_inj_perc_{sys}' for sys in systems if f'loc_inj_perc_{sys}' in df.columns]
    tract_cols = [f'tract_inj_perc_{sys}' for sys in systems if f'tract_inj_perc_{sys}' in df.columns]
    pre_systems = ['A4B2', 'M1', 'D1', 'D2', '5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6']
    pre_cols=[f'pre_{sys}' for sys in pre_systems if f'pre_{sys}' in df.columns]
    post_systems = ['VAChT', 'DAT', '5HTT']
    post_cols=[f'post_{sys}' for sys in post_systems if f'post_{sys}' in df.columns]
    if is_group or len(subjects) > 1:
        # Cas groupe : on calcule la moyenne
        mean_values = {}
        for col in loc_cols:
            mean_values[col] = plot_data[col].mean()
        for col in tract_cols:
            mean_values[col] = plot_data[col].mean()
        for col in pre_cols:
            mean_values[col]=plot_data[col].mean()
        for col in post_cols:
            mean_values[col]=plot_data[col].mean()
        data_to_plot = pd.Series(mean_values)
        
    else:
        # Cas sujet unique : on prend les données brutes
        data_to_plot = plot_data.iloc[0]  
    
    
    # 1. Préparation des données pour les graphiques 1 et 2
    loc_inj_perc = [data_to_plot[f'loc_inj_perc_{sys}'] for sys in systems]
    tract_inj_perc = [data_to_plot[f'tract_inj_perc_{sys}'] for sys in systems]

     # 2. Préparation des données pour le graphiques 3
    pre_cols_used=[data_to_plot[f'pre_{sys}'] for sys in pre_systems]
    post_cols_used=[data_to_plot[f'post_{sys}'] for sys in post_systems]
    radii3_log = pre_cols_used + post_cols_used

    # 2. Calcul des ratios pre/post comme NeuroTmap.py
    # def safe_get(data, system, prefix):
    #     col = f'{prefix}_{system}'
    #     return data[col] if col in data else 0.0
    
    # radii3a, radii3b = [], []
    # for i, sys in enumerate(pre_systems):
    #     recep = max(safe_get(data_to_plot, sys, 'loc_inj_perc'), safe_get(data_to_plot, sys, 'tract_inj_perc'))
    #     trans_sys = 'VAChT' if sys in ['A4B2', 'M1'] else 'DAT' if sys in ['D1', 'D2'] else '5HTT'
    #     trans = max(safe_get(data_to_plot, trans_sys, 'loc_inj_perc'), safe_get(data_to_plot, trans_sys, 'tract_inj_perc'))
        
    #     radii3a.append(trans / 0.1 if recep == 0 else trans / recep)
    #     radii3b.append(recep / 0.1 if trans == 0 else recep / trans)
    
    # radii3b_avg = [
    #     (radii3b[0] + radii3b[1]) / 2,
    #     (radii3b[2] + radii3b[3]) / 2,
    #     sum(radii3b[4:9]) / 5
    # ]
    
    # radii3 = np.append(radii3a, radii3b_avg)
    # radii3_log = np.where(radii3 == 0, -1, np.log(radii3))
    
    # Si overlay, définir une couleur unique -- peut être ajouté hachure transparente pour le rendu
    if is_overlay:
        if title_suffix not in st.session_state.overlay_color_map:
            hue = len(st.session_state.overlay_color_map) * 60 % 360
            color = f"hsla({hue}, 80%, 50%, 0.5)"
            st.session_state.overlay_color_map[title_suffix] = color
        overlay_color = st.session_state.overlay_color_map[title_suffix]

        # Définir une liste de couleurs transparentes pour le overlay
        colors1 = [overlay_color] * len(systems)
        colors3 = [overlay_color if np.exp(val) > 1 else overlay_color.replace("0.5", "0.2") for val in radii3_log]
      
    else:

        fixed_color_strong = 'lightskyblue'    # bleu pastel 
        fixed_color_light = 'rgba(135, 206, 250, 0.3)' # même bleu avec 0.3 d'opacité 

        colors1 = [fixed_color_strong] * len(systems)
        colors3 = [fixed_color_strong if np.exp(val) > 1 else fixed_color_light for val in radii3_log]
      
        #configuration de base dans NeuroTmap pour un seul sujet
        # colors1 = ["#B7B3D7", "#928CC1", "#6E66AD", "#B7DEDA", "#92CEC8", "#6BBDB5", 
        #         "#EBA8B1", "#FCFCED", "#FBFAE2", "#F8F8D6", "#F8F6CB", "#F6F4BE", "#F5F2B3"]
        # colors3 = ['#42BDB5' if val > 1 else '#F5F2B3' for val in radii3]

    # Configuration commune
    config = {
        'title_x': 0.2,  # Centre les titres ajustable
        'title_font_size': 14,
        'polar': {
            'angularaxis': {
                'direction': 'clockwise',
                'rotation': 90,
            },
            'bargap': 0.1  
        }
    }

    base_val = 0 if is_overlay else None
    fig1 = go.Figure()
    fig2 = go.Figure()
    fig3 = go.Figure()

    # Graphique 1: Lésions
    fig1.add_trace(go.Barpolar(
        r=loc_inj_perc,
        theta=systems,
        marker_color=colors1[:len(systems)],
        name=title_suffix,
        hovertemplate='<b>%{theta}</b><br>%{r:.2f}%<extra></extra>',
        width=np.pi/4,
        base=base_val
    ))
    fig1.update_layout(
        title_text=f'<b>Receptor/transporter lesion', #: {title_suffix}</b>',
        polar_radialaxis_ticksuffix='%',
        height=600,
        showlegend=True,
        margin=dict(l=50, r=0, t=30, b=10),
        font=dict(size=12),
        **config
    )

    # Graphique 2: Déconnexions
    fig2.add_trace(go.Barpolar(
        r=tract_inj_perc,
        theta=systems,
        marker_color=colors1[:len(systems)],
        name=title_suffix,
        hovertemplate='<b>%{theta}</b><br>%{r:.2f}%<extra></extra>',
        width=np.pi/4,
        base=base_val 
    ))
    fig2.update_layout(
        title_text=f'<b>Receptor/transporter disconnection', #: {title_suffix}</b>',
        polar_radialaxis_ticksuffix='%',
        height=600,
        showlegend=True,
        margin=dict(l=50, r=0, t=30, b=10),
        font=dict(size=12),
        **config
    )
 
    # Graphique 3: Ratios
    fig3.add_trace(go.Barpolar(
        r=radii3_log,
        theta= [f"pre {sys}" for sys in pre_systems] + [f"post {sys}" for sys in post_systems],
        marker_color=colors3,
        name=title_suffix,
        hovertemplate='<b>%{theta}</b><br>%{r:.2f}<extra></extra>',
        width=np.pi/4,
        base=base_val
    ))
    fig3.update_layout(
        title_text=f'<b>Pre/post synaptic ratios (log scale)',#: {title_suffix}</b>',
        polar_radialaxis_range=[-1, 1],
        height=500,  
        showlegend=True,
        font=dict(size=12),
        **config
    )


    return fig1, fig2, fig3, colors1, colors3

def get_subjects_and_title(df, analysis_type, existing_subjects=None, is_overlay=False, context=""):
    """
    Gère la sélection des sujets et génère un titre descriptif avec des clés uniques
    
    Args:
        df: DataFrame contenant les données
        analysis_type: Type d'analyse ("Single subject", "By session and sex", etc.)
        existing_subjects: Liste des sujets à exclure (pour overlay)
        is_overlay: Booléen indiquant si c'est pour un overlay
        context: Chaîne supplémentaire pour rendre les clés uniques
        
    Returns:
        Tuple (liste des sujets, titre, sex, session)
    """
    if existing_subjects is None:
        existing_subjects = []
    
    # Création de préfixes/suffixes uniques pour les clés
    overlay_prefix = "overlay_" if is_overlay else ""
    key_suffix = f"_{context}" if context else ""
    base_key = f"{overlay_prefix}{analysis_type}{key_suffix}".replace(" ", "_").lower()
    
    subjects = []
    title_prefix = "Overlay " if is_overlay else "Base"
    
    # 1. Cas sujet unique
    if analysis_type == "Single subject":
        available_subjects = [s for s in df['subject'].unique() if s not in existing_subjects]
        selected = st.selectbox(
            f"Select {'overlay ' if is_overlay else ''}subject:",
            options=sorted(available_subjects),
            key=f"{base_key}_subject_select"
        )
        subjects = [selected]
        #return subjects, f"{title_prefix}Subject: {selected}"
        return subjects, f"{title_prefix}: {selected}", None, None

    # 2. Cas par session
    elif analysis_type == "By session and sex":
        session = st.selectbox(
            "Select session:",
            options=["V1", "V2", "V3"],
            key=f"{base_key}_session_select"
        )
        
        sex_filter = st.radio(
            f"Sex filter for {session}:",
            ["All", "Men only", "Women only"],
            horizontal=True,
            key=f"{base_key}_sex_filter_{session}"
        )
        
        # Filtrage initial par session
        session_subjects = df[df['subject'].str.contains(f"_ses-{session}")]['subject'].tolist()
        
        # Filtrage supplémentaire par sexe
        if sex_filter != "All":
            gender = "M" if sex_filter == "Men only" else "F"
            session_subjects = df[
                (df['subject'].isin(session_subjects)) & 
                (df['sex'] == gender)
            ]['subject'].tolist()
        
        # Exclusion des sujets existants
        subjects = [s for s in session_subjects if s not in existing_subjects]
        if is_overlay:
            subjects = [s for s in subjects if s not in st.session_state.get("overlay_subjects", [])]

        # Construction du titre
        title = f"{title_prefix}: Session {session}"
        if sex_filter != "All":
            title += f" ({sex_filter})"
            
        return subjects, title, sex_filter, session


    # 3. Cas filtre combiné -- a modifier si d'autres idées de filtre
    else:
        selected_sessions = st.multiselect(
            "Select sessions:",
            options=["V1", "V2", "V3"],
            default=["V1", "V2", "V3"],
            key=f"{base_key}_multisession"
        )
        
        selected_genders = st.multiselect(
            "Select genders:",
            options=["Men (M)", "Women (F)"],
            default=["Men (M)", "Women (F)"],
            key=f"{base_key}_multigender"
        )
        
        # Conversion des genres en codes
        gender_codes = ["M" if g == "Men (M)" else "F" for g in selected_genders]
        
        # Filtrage combiné
        subjects = []
        for session in selected_sessions:
            for gender in gender_codes:
                group = df[
                    (df['subject'].str.contains(f"_ses-{session}")) &
                    (df['sex'] == gender)
                ]
                subjects.extend([s for s in group['subject'].unique() if s not in existing_subjects])
        if is_overlay:
            subjects = [s for s in subjects if s not in st.session_state.get("overlay_subjects", [])]

    
        # Construction du titre
        title = f"{title_prefix}:  "
        title += f"{', '.join(selected_sessions)} sessions"
        title += f", {', '.join(selected_genders)}"
        
        return subjects, title, None, None

def detect_group(subject_id):
    if "_sub-NA" in subject_id or "-NA" in subject_id:
        return "NA"
    elif "_sub-A" in subject_id or "-A" in subject_id:
        return "A"
    #pour les sujets controles pas possible dans NeuroTmap mais peut etre permettre comparaison des scores cliniques uniquement
    elif "_sub-C" in subject_id or "-C" in subject_id: 
        return "C"
    elif "_sub-AN" in subject_id or "-AN" in subject_id: 
        return "AN"
    elif "_sub-B" in subject_id or "-B" in subject_id: 
        return "B"
    elif "_sub-W" in subject_id or "-W" in subject_id: 
        return "W"
    elif "_sub-G" in subject_id or "-G" in subject_id: 
        return "G"
    else:
        return "Unknown"

def clear_overlay(df=None, subjects=None, plot_title=None):
                st.session_state.overlay_plots = None
                st.session_state.overlay_subjects = []
                st.session_state.overlay_title = ""
                st.session_state.overlay_ready = False

                # Si on veut regénérer les graphes de base
                if df is not None and subjects and plot_title:
                    with st.spinner("Resetting to base profile..."):
                        fig1, fig2, fig3, _, _ = create_interactive_plots(df, subjects, plot_title)
                        st.session_state.base_plots = (fig1, fig2, fig3)

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

def extract_pseudo_r2_cs_from_summary(model):
    summary_text = model.summary().as_text()
    match = re.search(r"Pseudo R-squ\. \(CS\):\s+([\d.]+)", summary_text)
    if match:
        return float(match.group(1))
    return None

def get_family_and_link(dist_name, link_name, var_power=None):
    # Map link functions
    link_map = {
        "log": Log(),
        "identity": Identity(),
        "inverse": InversePower(),
        "sqrt": Sqrt()
    }
    link_func = link_map.get(link_name, Log())  # Log par défaut

    # Map families
    if dist_name == "Gaussian":
        family = Gaussian(link=link_func)
    elif dist_name == "Gamma":
        family = Gamma(link=link_func)
    elif dist_name == "Poisson":
        family = Poisson(link=link_func)
    elif dist_name == "Tweedie":
        power = var_power if var_power is not None else 1.5
        family = Tweedie(var_power=power, link=link_func)
    else:
        family = Gaussian(link=link_func)

    return family

def check_model_assumptions(df, outcome, predictors, family):
    # Vérifier les valeurs manquantes
    missing = df[[outcome] + predictors].isna().sum()
    if missing.any():
        st.warning(f"Données manquantes :\n{missing[missing > 0]}")
    
    # Vérifier les valeurs pour la famille choisie
    if isinstance(family, (Poisson, Gamma)) and (df[outcome] <= 0).any():
        st.error(f"La variable {outcome} contient des valeurs ≤0 - incompatible avec {family.__class__.__name__}")
        return False
    
    # Vérifier la variance pour Poisson
    if isinstance(family, Poisson):
        if df[outcome].var() > df[outcome].mean() * 1.5:
            st.warning("Surdispersion détectée - envisagez Tweedie") #ou NegativeBinomial
    
    return True

def safe_glm_crossgroup(
    df_predictors,
    df_outcomes,
    outcomes,
    systems,
    covariate=[],
    visit_name="",
    family=None,
    interaction_var=None  # ex: 'Sexe_bin', 'Group', or None
):
    results = []

    for outcome in outcomes:
        outcome_var = f"Q('{outcome}')" if any(c.isdigit() for c in outcome) else outcome

        for system, predictors in systems.items():
            for predictor in predictors:
                formula_terms = []
                term = f"Q('{predictor}')" if any(c.isdigit() for c in predictor) else predictor
                formula_terms.append(term)

                if interaction_var and interaction_var in df_predictors.columns:
                    formula_terms.append(f"{term}:{interaction_var}")

                for cov in covariate:
                    if cov != interaction_var:
                        cov_term = f"Q('{cov}')" if any(c.isdigit() for c in cov) else cov
                        formula_terms.append(cov_term)

                formula = f"{outcome_var} ~ {' + '.join(formula_terms)}"
                #st.write("Formule GLM :", formula)

                try:
                    df_predictors_temp = df_predictors.reset_index() if 'subject' not in df_predictors.columns else df_predictors.copy()
                    df_outcomes_temp = df_outcomes.reset_index() if 'subject' not in df_outcomes.columns else df_outcomes.copy()

                    needed_cols = list(set(['subject', predictor] + covariate))
                    if interaction_var and interaction_var not in needed_cols:
                        needed_cols.append(interaction_var)

                    if outcome not in df_outcomes_temp.columns:
                        st.warning(f"{outcome} non trouvé dans les outcomes.")
                        continue

                    df_merged = df_outcomes_temp[['subject', outcome]].merge(
                        df_predictors_temp[needed_cols],
                        on='subject',
                        how='inner'
                    )

                    #st.write(f"Taille après merge: {len(df_merged)} observations")
                    #st.write("Valeurs manquantes par colonne:", df_merged.isna().sum())
                    if df_merged.empty:
                        st.warning(f"Aucune donnée après merge pour {outcome} ~ {predictor}")
                        continue

                    drop_cols = list(set([outcome, predictor] + covariate + ([interaction_var] if interaction_var else [])))
                    df_clean = df_merged.dropna(subset=drop_cols)
                    #st.write(f"Taille après suppression des NaN: {len(df_clean)} observations")

                    if df_clean.empty or len(df_clean) < 3:
                        st.warning(f"Données insuffisantes pour {outcome} ~ {predictor}")
                        continue

                    non_numeric = df_clean[drop_cols].select_dtypes(exclude=['number']).columns.tolist()
                    if non_numeric:
                        st.warning(f"Variables non numériques pour {outcome} ~ {predictor}: {non_numeric}")
                        continue

                    if not check_model_assumptions(df_clean, outcome, [predictor] + covariate, family):
                        st.warning(f"Hypothèses non respectées pour {outcome} ~ {predictor}")
                        continue

                    try:
                        model = smf.glm(formula, data=df_clean, family=family).fit()
                    except Exception as e:
                        error_msg = f"Erreur lors du fit du modèle pour {outcome} ~ {predictor}: {str(e)}"
                        if "endog" in str(e) and "log" in str(family.link):
                            error_msg += "\n⚠️ Essayez une autre famille/link (ex: valeurs négatives avec link='log')"
                        st.error(error_msg)
                        continue

                    n_obs = int(model.nobs)
                    df_resid = int(model.df_resid)
                    df_model = int(model.df_model)
                    log_likelihood = model.llf
                    deviance = model.deviance
                    pearson_chi2 = model.pearson_chi2
                    pseudo_r2 = extract_pseudo_r2_cs_from_summary(model)
                    scale = model.scale

                    for param in model.params.index:
                        coef = model.params[param]
                        pval = model.pvalues[param]
                        is_interaction = ':' in param
                        base_pred = param.split(':')[0].replace("Q('", "").replace("')", "")

                        results.append({
                            'Visit': visit_name,
                            'Outcome': outcome,
                            'System': system,
                            'Predictor': param,
                            'Base_Predictor': base_pred,
                            'Coefficient': coef,
                            'Effect_Type': 'Interaction' if is_interaction else 'Main',
                            'P-value': pval,
                            'Significant': pval < 0.05,
                            'N_obs': n_obs,
                            'Df_resid': df_resid,
                            'Df_model': df_model,
                            'Log-likelihood': log_likelihood,
                            'Deviance': deviance,
                            'Pearson_chi2': pearson_chi2,
                            'Pseudo_R2_CS': pseudo_r2,
                            'Scale': scale,
                        })

                except Exception as e:
                    print(f"Erreur avec {outcome} ~ {predictor}: {e}")
                    continue
    return pd.DataFrame(results)

def perform_group_comparison(group1_data, group2_data, paired=False):
    """
    Effectue une comparaison statistique entre deux groupes avec vérifications préalables
    
    Args:
        group1_data (pd.Series): Données du groupe 1
        group2_data (pd.Series): Données du groupe 2
        paired (bool): Si True, utilise un test apparié quand V1 ; V2 ; V3 comparaison
        
    Returns:
        dict: Dictionnaire contenant tous les résultats statistiques
    """
    # Nettoyage des données
    vals1 = group1_data.dropna()
    vals2 = group2_data.dropna()
    
    # Vérification des effectifs
    if len(vals1) < 3 or len(vals2) < 3:
        return None
    
    if paired and len(vals1) != len(vals2):
        raise ValueError("For paired tests, group sizes must be equal")

    results = {
        'n_group1': len(vals1),
        'n_group2': len(vals2),
        'mean_group1': vals1.mean(),
        'mean_group2': vals2.mean(),
        'std_group1': vals1.std(),
        'std_group2': vals2.std()
    }
    
    # 1. Test de normalité (Shapiro-Wilk)
    shapiro1 = shapiro(vals1)
    shapiro2 = shapiro(vals2)
    results.update({
        'shapiro_p1': shapiro1.pvalue,
        'shapiro_p2': shapiro2.pvalue,
        'normal_dist': (shapiro1.pvalue > 0.05) and (shapiro2.pvalue > 0.05)
    })
    
    # 2. Test d'homogénéité des variances (Levene) - seulement si non apparié
    if not paired:
        levene_test = levene(vals1, vals2)
        results.update({
            'levene_p': levene_test.pvalue,
            'equal_var': levene_test.pvalue > 0.05
        })
    
    # Choix du test statistique
    if paired:
        # Tests pour données appariées
        if results['normal_dist']:
            test_result = ttest_rel(vals1, vals2)
            results.update({
                'test_type': 'Paired t-test',
                'statistic': test_result.statistic,
                'p_value': test_result.pvalue,
                'effect_size': (vals1.mean() - vals2.mean()) / np.sqrt((vals1.std()**2 + vals2.std()**2)/2)
            })
        else:
            test_result = wilcoxon(vals1, vals2)
            results.update({
                'test_type': 'Wilcoxon signed-rank',
                'statistic': test_result.statistic,
                'p_value': test_result.pvalue,
                'effect_size': test_result.statistic / np.sqrt(len(vals1))
            })
    else:
        # Tests pour groupes indépendants
        if results['normal_dist']:
            if results.get('equal_var', True):
                test_result = ttest_ind(vals1, vals2, equal_var=True)
                test_type = "Student-t (var égales)"
            else:
                test_result = ttest_ind(vals1, vals2, equal_var=False)
                test_type = "Welch-t (var inégales)"
            
            effect_size = (vals1.mean() - vals2.mean()) / np.sqrt((vals1.std()**2 + vals2.std()**2)/2)
            
            try:
                analysis = TTestIndPower()
                power = analysis.power(
                    effect_size=effect_size, 
                    nobs1=len(vals1), 
                    alpha=0.05,
                    ratio=len(vals2)/len(vals1), 
                    alternative='two-sided'
                )
            except:
                power = np.nan
                
            results.update({
                'test_type': test_type,
                'statistic': test_result.statistic,
                'p_value': test_result.pvalue,
                'effect_size': effect_size,
                'power': power
            })
        else:
            test_result = mannwhitneyu(vals1, vals2, alternative='two-sided')
            n1, n2 = len(vals1), len(vals2)
            U = test_result.statistic
            Z = (U - n1*n2/2) / np.sqrt(n1*n2*(n1+n2+1)/12)  # Conversion U → Z
            p_value_from_z = 2 * (1 - norm.cdf(abs(Z)))
            effect_size = Z / np.sqrt(n1 + n2)
            
            results.update({
                'test_type': "Mann-Whitney U",
                'statistic': test_result.statistic,
                'p_value': test_result.pvalue,
                'statistic_z': Z,
                'p_value_from_z': p_value_from_z,
                'effect_size': effect_size,
                #'power': np.nan
            })
    
    results['significant'] = results['p_value'] < 0.05
    return results

def clean_groups_for_variable(df1, df2, var, paired):
    """Supprime les sujets ayant des valeurs manquantes pour une variable sélectionné par l'utilsateur.
       Si paired, conserve uniquement les paires valides."""
    df1_valid = df1[df1[var].notna()]
    df2_valid = df2[df2[var].notna()]

    if paired:
        # On garde les sujets présents et valides dans les deux groupes
        base_ids_1 = df1_valid['subject'].apply(lambda x: x.split('-V')[0])
        base_ids_2 = df2_valid['subject'].apply(lambda x: x.split('-V')[0])
        common_bases = set(base_ids_1).intersection(set(base_ids_2))

        df1_clean = df1_valid[df1_valid['subject'].apply(lambda x: x.split('-V')[0]).isin(common_bases)]
        df2_clean = df2_valid[df2_valid['subject'].apply(lambda x: x.split('-V')[0]).isin(common_bases)]

        return df1_clean, df2_clean, len(common_bases)
    else:
        return df1_valid, df2_valid, None

def get_correlation_matrix(df, include_sex_bin=True):
    """
    Calculate correlation matrix with automatic test selection (Pearson/Spearman)
    and FDR correction for multiple comparisons.
    
    Parameters:
    - df: DataFrame containing the variables to correlate
    - include_sex_bin: Whether to include 'Sexe_bin' in the matrix (False for single-sex analysis)
    
    Returns:
    - corr_matrix: DataFrame of correlation coefficients
    - pval_matrix: DataFrame of FDR-corrected p-values
    """
    # Select and clean numeric data
    df_num = df.select_dtypes(include=['float64', 'int64','bool']).dropna(axis=1, thresh=int(0.5 * len(df)))
    
    # Convert bool to int if needed
    df_num = df_num.apply(lambda x: x.astype(int) if x.dtype == bool else x)

    # Exclude Sexe_bin if requested
    if not include_sex_bin and 'Sexe_bin' in df_num.columns:
        df_num = df_num.drop(columns=['Sexe_bin'])
        
    cols = df_num.columns
    corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pval_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    pvals_list = []
    index_pairs = []

    for col1, col2 in combinations(cols, 2):
        x, y = df_num[col1].dropna(), df_num[col2].dropna()
        common_index = x.index.intersection(y.index)
        x, y = x.loc[common_index], y.loc[common_index]

        if len(x) < 3:
            continue  # Skip pairs with less than 3 observations

        # Normality test with error handling
        try:
            norm_x = shapiro(x)[1] > 0.05
            norm_y = shapiro(y)[1] > 0.05
        except:
            norm_x = norm_y = False

        # Choose appropriate correlation test
        if norm_x and norm_y:
            corr, pval = pearsonr(x, y)
        else:
            corr, pval = spearmanr(x, y)

        # Store results
        corr_matrix.loc[col1, col2] = corr
        corr_matrix.loc[col2, col1] = corr
        pval_matrix.loc[col1, col2] = pval
        pval_matrix.loc[col2, col1] = pval
        
        # Prepare for FDR correction
        pvals_list.append(pval)
        index_pairs.append((col1, col2))

    # Fill diagonal
    np.fill_diagonal(corr_matrix.values, 1.0)
    for col in cols:
        pval_matrix.loc[col, col] = 0.0  # p-value for diagonal is 0

    # Apply FDR correction (Benjamini-Hochberg)
    if pvals_list:
        _, pvals_corrected, _, _ = multipletests(pvals_list, alpha=0.05, method='fdr_bh')
        for (col1, col2), p_corr in zip(index_pairs, pvals_corrected):
            pval_matrix.loc[col1, col2] = p_corr
            pval_matrix.loc[col2, col1] = p_corr
    
    return corr_matrix, pval_matrix

def extract_subject_id(subject_name):
                match = re.match(r"(sub-[A-Za-z0-9]+)_ses-V[0-9]+", subject_name)
                return match.group(1) if match else None

def clean_predictor_name(name):
                    # Enlève Q(' et ') si présent
                    if isinstance(name, str) and name.startswith("Q('") and name.endswith("')"):
                        return name[3:-2]
                    return name

def style_corr_with_pval(corr_df, pval_df, p_thresh):
    def highlight(val, pval):
        if pval >= p_thresh:
            # Griser les cases en conservanr la valeur
            return 'color: lightgray;'
        else:
            return ''
# ------------------------ Streamlit app ---------------------------------------------------------------
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #e6f1fb;
    }
    .custom-container {
        background-color: #eaf4fc;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 30px;
    }

    .custom-title {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-size: 46px;
        font-weight: 600;
        color: #1f4e79;
        margin-bottom: 5px;
        border-bottom: 3px solid transparent;
        display: inline-block;
        animation: underline-slide 3s ease-in-out infinite alternate;
    }

    @keyframes underline-slide {
        0%   { border-color: #1f4e79; }
        100% { border-color: #82c0ff; }
    }

    .custom-subtitle {
        font-size: 20px;
        color: #1f4e79;
        margin-top: 5px;
        font-weight: 400;
    }
    </style>

    <div class="custom-container">
        <div class="custom-title">
            🧠 Neurotransmitter Balance & Outcomes
        </div>
        <div class="custom-subtitle">
            Explore neurotransmitter ratios and their clinical relevance
        </div>
    </div>
    """,
    unsafe_allow_html=True
    )


# 1. Upload
uploaded_zip = st.file_uploader(
    "📦 **Upload a ZIP file containing all patient CSVs obtained with NeuroTmap.py and a clinical_data file** ",
    type=["zip"]
)

st.caption(
    "**The ZIP file must include:**\n"
    "- One or more `output_les_dis_sub-XXX_ses-VX.csv` files\n"
    "- One or more `output_pre_post_synaptic_ratio_sub-XXX_ses-VX.csv` files\n"
    "- *(Optional)* One `clinical_data.csv` or `.xlsx` file\n\n"
    "**Important:**\n"
    "- Subject IDs in all files must follow the format: `sub-<group letter>XXX-...`\n"
    "  - Group letters: NA (Non-aphasic), A (Aphasic), G (Global aphasia), W (Wernicke aphasia), B(Broca aphasia), C (Conduction aphasia), AN (Anomic aphasia)\n"
    "- The `clinical_data` file must include a `subject` column **exactly matching** the IDs in the filenames.\n"
    "- You may include other columns such as: `sex`, `timepoint`, `repetition_score`, "
    "`comprehension_score`, `naming_score`, `composite_score`, `lesion_volume`.\n"
    "- Lesion volume must be in **mm³**."
    # """
    # The ZIP file must include:
    # - One or more `output_les_dis_sub-XXX_ses-VX.csv` files
    # - One or more `output_pre_post_synaptic_ratio_sub-XXX_ses-VX.csv` files
    # - Optional one `clinical_data.csv` or `.xlsx` file  
    #   This clinical file **must include a `subject` column** matching the filenames, and can include other variables like:
    #   `sex`, `timepoint`, `repetition_score`, `comprehension_score`, `naming_score`, `composite_score`,  `lesion_volume`
    #   Lesion volume must be in mm3.
    # """
)

if uploaded_zip is not None:
    with st.spinner("⏳ Processing..."):
        df_combined = process_zip_to_dataframe(uploaded_zip)
    if not df_combined.empty:
        st.success("✅ All data combined successfully!")
        if st.checkbox("Show full combined dataset"):
            st.dataframe(df_combined)  
    else:
        st.warning("❌ No data could be combined. Please check the filenames or their contents.")

#Interface principale : visualisation des graphiques de neuroTmap

if uploaded_zip is not None and not df_combined.empty:
    # Initialisation du session state
    if 'base_plots' not in st.session_state:
        st.session_state.base_plots = None
    if 'overlay_plots' not in st.session_state:
        st.session_state.overlay_plots = None
    if 'show_overlay' not in st.session_state:
        st.session_state.show_overlay = False
    if 'overlay_subjects' not in st.session_state:
        st.session_state.overlay_subjects = []
    if "overlay_color_map" not in st.session_state:
        st.session_state.overlay_color_map = {} 

    st.header("📊 Neurotransmitter Profile Visualization")
    
    # Colonnes pour l'interface
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Sélection du groupe principal
        st.subheader("Main Profile Selection")
        analysis_type = st.radio(
            "Analysis type:",
            ["Single subject", "By session and sex", "Combined filter"],
            horizontal=True
        )
        
        #doit enlever le detect_group si "Single Subject" -- crée un bug
        subjects, plot_title, sex_filter, session = get_subjects_and_title(df_combined, analysis_type,context="main")
        if analysis_type != "Single subject":
            st.write(
                "Select the subject groups to include in the analysis. "
                "If you do not want to include a group, simply **uncheck** it.\n\n"
                "**Groups:**\n"
                "- NA = Non-aphasic\n"
                "- A = Aphasic\n"
                "- G = Global aphasia\n"
                "- W = Wernicke aphasia\n"
                "- B = Broca aphasia\n"
                "- C = Conduction aphasia\n"
                "- AN = Anomic aphasia"
            )
            subject_groups = {
                subj: detect_group(subj) for subj in subjects
            }
            
            available_groups = sorted(set(subject_groups.values()))
            selected_groups = st.multiselect(
                "Filter by subject group:",
                options=available_groups,
                default=available_groups,
                key="group_filter"
            )
            # Filtrer les sujets en fonction des groupes sélectionnés
            subjects = [subj for subj in subjects if subject_groups[subj] in selected_groups]
        st.write(f"Number of subjects in main profile: {len(subjects)}")
        # Bouton de génération
        if st.button("Generate Main Profile"):
            with st.spinner("Generating main profile..."):
                is_group = analysis_type != "Single subject"
                fig1, fig2, fig3, colors1, colors3 = create_interactive_plots(df_combined, subjects, plot_title, is_group=is_group)
                st.session_state.base_plots = (fig1, fig2, fig3)
                st.session_state.show_overlay = False  # Reset overlay quand on change le main 
                st.rerun()

    with col2:
        # 2. Options d'affichage et overlay
        st.subheader("Display Options")
        show_data = st.checkbox("Show selected data", value=False)

        # Afficher les données si demandé
        if show_data:
            st.subheader("Selected Subjects Data")
            overlay_subjects = st.session_state.get("overlay_subjects", [])
            all_subjects = list(set(subjects) | set(overlay_subjects))
            st.dataframe(df_combined[df_combined['subject'].isin(all_subjects)])
        
        prev_show_overlay = st.session_state.get("show_overlay", False)
        show_overlay = st.checkbox(
            "Enable overlay comparison", 
            value=prev_show_overlay,
            key="enable_overlay"
        )

        # Si l'utilisateur a décoché l'overlay, on efface tout
        if prev_show_overlay and not show_overlay:
            clear_overlay(df_combined, subjects, plot_title)
            st.session_state.show_overlay = False
            st.rerun()

        # Met à jour l'état final
        st.session_state.show_overlay = show_overlay
        
        # Si overlay activé, afficher les options de sélection
        if st.session_state.show_overlay:
            st.markdown("---")
            st.subheader("Overlay Profile Selection")
            
            overlay_type = st.radio(
                "Overlay type:",
                ["Single subject", "By session and sex", "Combined filter"],
                horizontal=True,
                key="overlay_type"
            )

            if "current_overlay_selection" not in st.session_state:
                st.session_state.current_overlay_selection = []
            # Logique de sélection des sujets overlay
            overlay_subjects, overlay_title, sex_filter, session = get_subjects_and_title(
                df_combined,
                overlay_type, 
                existing_subjects=subjects,
                is_overlay=True,
                context="overlay"
            )
            if overlay_type != "Single subject":
                overlay_subject_groups = {
                    subj: detect_group(subj) for subj in overlay_subjects
                }
                
                available_ov_groups = sorted(set(overlay_subject_groups.values()))
                selected_ov_groups = st.multiselect(
                    "Filter overlay by subject group:",
                    options=available_ov_groups,
                    default=available_ov_groups,
                    key="overlay_group_filter"
                )
                # Filtrer les sujets en fonction des groupes sélectionnés
                overlay_subjects = [subj for subj in overlay_subjects if overlay_subject_groups[subj] in selected_ov_groups]

            if "overlay_ready" not in st.session_state:
                st.session_state.overlay_ready = False
            
            # Bouton Apply Overlay
            if st.button("Apply Overlay"):
                with st.spinner("Generating  overlay profile..."):
                    # Met à jour la sélection courante avec la sélection active dans le widget
                    is_group = overlay_type != "Single subject"
                    current_overlay = st.session_state.get("overlay_subjects", [])
                    new_subjects = [s for s in overlay_subjects if s not in current_overlay]
                    if is_group:
                        if set(new_subjects) != set(current_overlay):
                            fig1_ov, fig2_ov, fig3_ov, _, _ = create_interactive_plots(
                                df_combined,
                                new_subjects,
                                title_suffix=overlay_title,
                                is_group=True,
                                is_overlay=True
                            )
                            st.session_state.overlay_plots = (fig1_ov, fig2_ov, fig3_ov)
                            st.session_state.overlay_subjects = new_subjects  
                            st.session_state.overlay_title = overlay_title
                            st.session_state.overlay_ready = True
                            st.write(f"Overlay applied for group with {len(new_subjects)} subjects.")
                        else:
                            st.info("Selected group already in overlay.")
                    else:
                        # Pour sujets individuels, on ajoute sujet par sujet
                        if new_subjects:
                            for subj in new_subjects:
                                fig1_ov, fig2_ov, fig3_ov, _, _ = create_interactive_plots(
                                    df_combined,
                                    [subj],
                                    title_suffix=overlay_title,
                                    is_group=False,
                                    is_overlay=True
                                )
                                if "overlay_plots" not in st.session_state or st.session_state.overlay_plots is None:
                                    st.session_state.overlay_plots = (fig1_ov, fig2_ov, fig3_ov)
                                    st.session_state.overlay_subjects = [subj]
                                else:
                                    st.session_state.overlay_plots = (fig1_ov, fig2_ov, fig3_ov)
                                    st.session_state.overlay_subjects = [subj]
                                st.session_state.overlay_title = overlay_title
                                st.session_state.overlay_ready = True
                            st.write(f"Overlay applied for {len(new_subjects)} subject(s).")
                        else:
                            st.info("Selected subject(s) already in overlay.")

            #bouton pour clear tous les sujets overlay
            
            if st.button("🗑️ Clear Overlay"):
                clear_overlay(df_combined, subjects, plot_title)
                st.rerun()

    # ------------------------ AFFICHAGE DES GRAPHIQUES ------------------------
    if "base_plots" in st.session_state and st.session_state.base_plots:
        fig1, fig2, fig3 = st.session_state.base_plots
        # Configurer le layout pour superposition avant d'ajouter les overlays
        for fig in [fig1, fig2, fig3]:
            fig.update_layout(barmode='overlay', bargap=0.2)

        if (
            st.session_state.show_overlay 
            and st.session_state.get("overlay_plots") 
            and st.session_state.get("overlay_ready", False)
        ):
            try:
                fig1_ov, fig2_ov, fig3_ov = st.session_state.overlay_plots
                for fig, fig_ov in zip([fig1, fig2, fig3], [fig1_ov, fig2_ov, fig3_ov]):
                    fig.add_traces(fig_ov.data)
            except (ValueError, TypeError) as e:
                st.warning(f"Erreur lors du chargement des overlays : {e}")
            finally:
                # Une fois les overlays appliqués, on reset le flag
                st.session_state.overlay_ready = False     
        # Afficher les graphiques
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            st.plotly_chart(fig3, use_container_width=True)
            
            



    # ------------------------Analyses statistiques ------------------------

   
    st.header("📈 Statistical Analysis")

    analysis_method = st.selectbox(
        "Choose analysis method:",
        options=["GLM", "T-Test", "Correlation"]
    )

    if analysis_method == "GLM":
        dist = st.selectbox("Select distribution:", ["Gaussian", "Gamma", "Poisson", "Tweedie"])
        if dist == "Tweedie":
            var_power = st.number_input("Variance power", min_value=0.0, max_value=3.0, value=1.5, step=0.1)
  
        interaction = st.checkbox("Include interaction", value=False)
        fam_to_links = {
            "Gaussian": ["identity", "log", "inverse"],
            "Gamma": ["identity", "log", "inverse"],
            "Poisson": ["log", "identity"],
            "Tweedie": ["log", "identity", "inverse"],
        }
        link_map = {
            "log": Log(),
            "identity": Identity(),
            "inverse": InversePower(),
            "sqrt": Sqrt()
        }
        valid_links = fam_to_links[dist]
        link = st.selectbox("Select link function:", options=valid_links)
        selected_link = link_map[link]

    elif analysis_method == "T-Test":
        st.markdown("T-Test options will be handled automatically based on normality and variance homogeneity.")

    elif analysis_method == "Correlation":
        st.markdown("Correlation test will be chosen automatically (Pearson if normal, Spearman otherwise).")

    # ==== Section filtre sur les sujets ====
    if analysis_method == "Correlation":
        analysis_type = st.radio(
                    "Analysis type:",
                    ["By session and sex", "Personalized subject list"],
                    horizontal=True
                )
    else : 
        analysis_type = st.radio(
                    "Analysis type:",
                    ["By session and sex", "Combined filter", "Personalized subject list"],
                    horizontal=True
                )
        
    if analysis_type != "Personalized subject list":

        #bug avec "sex_bin" quand include interaction
        if analysis_method=="GLM":
            subjects, title, sex_filter, session = get_subjects_and_title(
                df=df_combined,
                analysis_type=analysis_type,
                existing_subjects=[],
                is_overlay=False,
                context="stats"
            )
            subject_groups = {
                    subj: detect_group(subj) for subj in subjects
                }
                
            available_groups = sorted(set(subject_groups.values()))
            selected_groups = st.multiselect(
                "Filter by subject group:",
                options=available_groups,
                default=available_groups,
                key="group_filter_2"
            )
            # Filtrer les sujets en fonction des groupes sélectionnés
            subjects = [subj for subj in subjects if subject_groups[subj] in selected_groups]

        elif analysis_method == "T-Test": 
            st.write("#### Group 1 Selection")
            # Sélection du groupe 1
            group1_subjects, group1_title, _, _ = get_subjects_and_title(
                df=df_combined,
                analysis_type=analysis_type,
                existing_subjects=[],
                is_overlay=False,
                context="stats_group1"
            )
            
            # Filtre par groupe pour le groupe 1
            group1_subject_groups = {
                subj: detect_group(subj) for subj in group1_subjects
            }
            available_groups_group1 = sorted(set(group1_subject_groups.values()))
            selected_groups_group1 = st.multiselect(
                "Filter Group 1 by subject group:",
                options=available_groups_group1,
                default=available_groups_group1,
                key="group_filter_group1"
            )
            group1_subjects = [subj for subj in group1_subjects if group1_subject_groups[subj] in selected_groups_group1]
            
            st.write("---")
            st.write("#### Group 2 Selection")
            # Sélection du groupe 2 (en excluant les sujets déjà sélectionnés dans le groupe 1)
            group2_subjects, group2_title, _, _ = get_subjects_and_title(
                df=df_combined,
                analysis_type=analysis_type,
                existing_subjects=group1_subjects,  # Exclure les sujets du groupe 1
                is_overlay=False,
                context="stats_group2"
            )
            
            # Filtre par groupe pour le groupe 2
            group2_subject_groups = {
                subj: detect_group(subj) for subj in group2_subjects
            }
            available_groups_group2 = sorted(set(group2_subject_groups.values()))
            selected_groups_group2 = st.multiselect(
                "Filter Group 2 by subject group:",
                options=available_groups_group2,
                default=available_groups_group2,
                key="group_filter_group2"
            )
            group2_subjects = [subj for subj in group2_subjects if group2_subject_groups[subj] in selected_groups_group2]
            
            # Affichage du nombre de sujets sélectionnés
            st.write(f"Number of subjects in Group 1: {len(group1_subjects)}")
            st.write(f"Number of subjects in Group 2: {len(group2_subjects)}")
            
            # Vérification qu'il y a assez de sujets
            if len(group1_subjects) < 3 or len(group2_subjects) < 3:
                st.warning("Each group must contain at least 3 subjects for statistical comparison")

        elif analysis_method=="Correlation":
            # Colonnes pour la sélection
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔵 Set 1 Configuration")
                group1_subjects, group1_title, _, session1 = get_subjects_and_title(
                    df=df_combined,
                    analysis_type=analysis_type,
                    existing_subjects=[],
                    is_overlay=False,
                    context="corr_group1"
                )

                group1_subject_groups = {
                    subj: detect_group(subj) for subj in group1_subjects
                }
                available_groups_group1 = sorted(set(group1_subject_groups.values()))
                selected_groups_group1 = st.multiselect(
                    "Filter Group 1 by subject group:",
                    options=available_groups_group1,
                    default=available_groups_group1,
                    key="group_filter_group1_corr"
                )
                group1_subjects = [subj for subj in group1_subjects if group1_subject_groups[subj] in selected_groups_group1]

            with col2:
                st.markdown("#### 🔴 Set 2 Configuration")
                group2_subjects, group2_title, _ , session2= get_subjects_and_title(
                    df=df_combined,
                    analysis_type=analysis_type,
                    existing_subjects=[],
                    is_overlay=False,
                    context="corr_group2"
                )

                group2_subject_groups = {
                    subj: detect_group(subj) for subj in group2_subjects
                }
                available_groups_group2 = sorted(set(group2_subject_groups.values()))
                selected_groups_group2 = st.multiselect(
                    "Filter Group 2 by subject group:",
                    options=available_groups_group2,
                    default=available_groups_group2,
                    key="group_filter_group2_corr"
                )
                group2_subjects = [subj for subj in group2_subjects if group2_subject_groups[subj] in selected_groups_group2]

    else:
        # Choix manuel de plusieurs sujets parmi tous (pas de filtre session/sex)
        st.caption(
    """
    The subject selection must include at least 3 subjects by group"""
    )   
        if analysis_method=="GLM":
            all_subjects = sorted(df_combined['subject'].unique())
            selected_subjects = st.multiselect(
                "Select personalized subjects:",
                options=all_subjects,
                key="personalized_subjects_select"
            )
            subjects = selected_subjects

        elif analysis_method=="Correlation":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔵 Set 1 Configuration")
                all_subjects = sorted(df_combined['subject'].unique())
                # Sélection Groupe 1 (sans exclusion)
                group1_subjects = st.multiselect(
                    "Select group 1 subjects",
                    options=all_subjects,
                    key="corr_group1_select"
                )
                session1 = "combined"
            with col2:
                st.markdown("#### 🔴 Set 2 Configuration")  
                all_subjects = sorted(df_combined['subject'].unique())  
                # Sélection Groupe 2 (PEUT inclure les mêmes sujets que groupe 1)
                group2_subjects = st.multiselect(
                    "Select group 2 subjects",
                    options=all_subjects,  # Pas de filtrage des sujets du groupe 1
                    key="corr_group2_select"
                )
                session2 = "combined"
        
    if analysis_type == "By session and sex" and sex_filter == "All":
        # Supprimer les colonnes liées au sexe si elles existent sinon échec de GLM
        df_combined = df_combined.drop(columns=[col for col in ["sex", "Sexe_bin"] if col in df_combined.columns])

   
    if analysis_method == "GLM":
        # Initialisation du session state
        if 'glm_stat' not in st.session_state:
            st.session_state.glm_stat = {
                'ran': False,
                'results': None,
                'variables': {
                    'outcomes': [],
                    'covariates': [],
                    'predictors': []
                },
                'plot_config': {
                    'selected_var': None,
                    'show_points': True,
                    'color_by': "None",
                    'figure': None,
                    'last_run_variable': None
                },
                'analysis_done': False,
                'data_ready': False
            }

        # Filtrer le dataframe avec la liste des sujets sélectionnés
        df_filtered = df_combined[df_combined['subject'].isin(subjects)]
        # Conversion du sexe en binaire 
        sex_col = next((col for col in ['Sexe', 'sex', 'gender', 'Sex', 'sexe'] if col in df_filtered.columns), None)
        if sex_col and 'Sexe_bin' not in df_filtered.columns:
            sex_mapping = {
                'm': 0, 'male': 0, 'homme': 0, 'man': 0, 'men': 0,
                'f': 1, 'female': 1, 'femme': 1, 'woman': 1, 'women': 1
            }
            df_filtered['Sexe_bin'] = df_filtered[sex_col].astype(str).str.strip().str.lower().map(sex_mapping)
            
            if df_filtered['Sexe_bin'].isna().any():
                st.warning("Certaines valeurs de sexe n'ont pas pu être converties")

        st.write(f"Filtered subjects count: {len(subjects)}")
        st.write(" Data filtered before the model:", df_filtered)
        st.write("DataFrame size:", df_filtered.shape)
        
        # Re-définir la liste des systèmes 
        systems = [col.replace('loc_inj_', '') for col in df_combined.columns 
                if col.startswith('loc_inj_') and not col.startswith('loc_inj_perc')]

        # Séparer les colonnes cliniques (variables dépendantes) 
        clinical_cols = [col for col in df_combined.columns if col not in ['subject','Sexe_bin','sex']
                        and not any(sys in col for sys in systems)]

        # Séparer les prédicteurs liés aux systèmes
        system_predictors = [col for col in df_combined.columns
                            if any(sys in col for sys in systems)]

       
        # Variables : Pre/Post-synaptic ratios
        pre_post_vars = [var for var in [
            "pre_A4B2", "pre_M1", "pre_D1", "pre_D2",
            "pre_5HT1a", "pre_5HT1b", "pre_5HT2a",
            "pre_5HT4", "pre_5HT6",
            "post_VAChT", "post_DAT", "post_5HTT"
        ] if var in system_predictors]

        # Variables : neurotransmitter systems (local & tract)
        nt_systems_vars_loc = [var for var in [
            "loc_inj_GABAa", "loc_inj_mGluR5", 
            "loc_inj_MU", "loc_inj_H3",
            "loc_inj_CB1",  "loc_inj_A4B2", 
            "loc_inj_M1",  "loc_inj_VAChT", 
            "loc_inj_D1", "loc_inj_D2", 
            "loc_inj_DAT",  "loc_inj_Nor", 
            "loc_inj_5HT1a",  "loc_inj_5HT1b", 
            "loc_inj_5HT2a",  "loc_inj_5HT4", 
            "loc_inj_5HT6", "loc_inj_5HTT"
        ] if var in system_predictors]

        nt_systems_vars_tract= [var for var in [
            "tract_inj_GABAa", "tract_inj_mGluR5",
            "tract_inj_MU", "tract_inj_H3",
            "tract_inj_CB1", "tract_inj_A4B2",
            "tract_inj_M1", "tract_inj_VAChT",
            "tract_inj_D1", "tract_inj_D2",
            "tract_inj_DAT",  "tract_inj_Nor",
            "tract_inj_5HT1a",  "tract_inj_5HT1b",
            "tract_inj_5HT2a",  "tract_inj_5HT4",
            "tract_inj_5HT6",  "tract_inj_5HTT"
        ] if var in system_predictors]


        # Partie 2 : Interface utilisateur pour la GLM
        st.subheader("🧠 GLM Variable Selection")

        selected_outcomes = st.multiselect(
            "Select clinical outcomes to predict (dependent variables):",
            options=clinical_cols,
            key="outcomes_selection",
            help="Choose one or more clinical/behavioral outcomes from your dataset."
        )
        # Filtrer les covariables disponibles (exclure celles déjà sélectionnées comme outcomes)
        available_covariates = [col for col in clinical_cols if col not in selected_outcomes and col!="sex"]

        if analysis_type == "By session and sex" :
            available_covariates=available_covariates
        else: 
            if 'Sexe_bin' not in available_covariates and 'Sexe_bin' in df_filtered.columns:
                available_covariates.append('Sexe_bin')

        selected_covariates = st.multiselect(
        "Select covariates for GLM:",
        options=available_covariates,
        default=[],
        key="covariate_selection"
        )

        system_options = {
            "Synaptic ratio": pre_post_vars,
            "Neurotransmitter (Loc)": nt_systems_vars_loc,
            "Neurotransmitter (Tract)": nt_systems_vars_tract
        }

        selected_system = st.radio(
            "Choose the predictor system:",
            list(system_options.keys()),
            key="system_selector"
        )

        # Récupération des variables associées
        selected_predictor = system_options[selected_system]

        interaction_vars = ['Sexe_bin', 'Group', None]
        if interaction: 
            selected_interaction = st.selectbox(
            "Select variable for interaction with predictors (or None):",
            options=interaction_vars,
            index=0,
            key="interaction_choice"
        )
        
        if dist == "Tweedie":
            var_power = var_power
            family = Tweedie(var_power=var_power, link=selected_link)
        elif dist == "Gamma":
            family = Gamma(link=selected_link)
        elif dist == "Poisson":
            family = Poisson(link=selected_link)
        else:
            family = Gaussian(link=selected_link)

        for outcome in selected_outcomes:
            outcome_values = df_filtered[outcome].dropna()
            if (outcome_values <= 0).any() and dist in ["Gamma", "Poisson"]:
                st.warning(f"⚠️ La variable {outcome} contient des valeurs négatives ou nulles - inapproprié pour {dist}")
            if (outcome_values % 1 != 0).any() and dist == "Poisson":
                st.warning(f"⚠️ La variable {outcome} contient des valeurs non entières - inapproprié pour Poisson")
        st.markdown("---")
        run_glm = st.button("🚀 Run GLM on selected outcomes")
        if run_glm:
            st.session_state.glm_stat['run_triggered'] = True
            st.session_state.glm_stat.update({
                    'variables': {
                        'outcomes': selected_outcomes,
                        'covariates': selected_covariates,
                        'predictors': selected_predictor
                    },
                    'ran': True,
                    'analysis_done': True,
                    'data_ready': True
                })
            st.session_state.df_filtered_glm = df_filtered.copy()

        if st.session_state.glm_stat.get('analysis_done') and st.session_state.glm_stat.get('data_ready'):
            df_filtered = st.session_state.get('df_filtered_glm', df_combined.copy())
            # Vérification préalable
            if len(subjects) < 3:
                st.error("⚠️ Vous devez sélectionner au moins 3 sujets pour l'analyse")
                st.stop()
                
            if not selected_outcomes:
                st.error("⚠️ Veuillez sélectionner au moins une variable dépendante")
                st.stop()
            if not selected_predictor:
                st.error("⚠️ Veuillez sélectionner au moins une variable indépendante")
                st.stop()
            previous_selected_var = None
            previous_show_points = None
            previous_color_by = None

            # Afficher un aperçu des données
            st.write("Preview of selected data:")
            st.write(df_filtered[selected_outcomes + selected_covariates + selected_predictor].describe())
        
            
            # Liste des variables numériques
            glm_config = st.session_state.glm_stat['plot_config']

            # Liste des variables numériques disponibles
            all_numeric_vars = [
                col for col in selected_outcomes + selected_covariates + selected_predictor
                if pd.api.types.is_numeric_dtype(df_filtered[col])
            ]

            if not all_numeric_vars:
                st.warning("⚠️ No numeric variable available for display.")
                st.stop()

            st.subheader("📊 Distribution of selected variables")

            selected_var = st.selectbox(
                "Choose a variable to visualize:",
                options=all_numeric_vars,
                index=all_numeric_vars.index(previous_selected_var) if previous_selected_var in all_numeric_vars else 0,
                key="glm_selected_var_box"
            )

            col1, col2 = st.columns(2)
            with col1:
                show_points = st.checkbox(
                    "Show individual points",
                    value=previous_show_points,
                    key="glm_show_points"
                )

            with col2:
                color_options = ["None"] + [col for col in ['Sexe_bin', 'Group'] if col in df_filtered.columns]               
                color_by = st.selectbox(
                    "Color by:",
                    options=color_options,
                    index=color_options.index(previous_color_by) if previous_color_by in color_options else 0,
                    key="glm_color_by"
                )
            
            previous_selected_var = glm_config.get('selected_var')
            previous_show_points = glm_config.get('show_points')
            previous_color_by = glm_config.get('color_by')

      
            if (
                selected_var != previous_selected_var
                or show_points != previous_show_points
                or color_by != previous_color_by
                or glm_config['figure'] is None
            ):

                if color_by != "None" and color_by in df_filtered.columns:
                    fig = px.box(
                        df_filtered,
                        y=selected_var,
                        x=color_by if len(df_filtered[color_by].unique()) > 1 else None,
                        color=color_by,
                        title=f"Distribution of {selected_var} by {color_by}",
                        points="all" if show_points else None,
                        color_discrete_map={0: '#3498db', 1: '#e74c3c'} if color_by == 'Sexe_bin' else None
                    )
                else:
                    fig = px.box(
                        df_filtered,
                        y=selected_var,
                        title=f"Distribution of  {selected_var}",
                        points="all" if show_points else None,
                        color_discrete_sequence=['#3498db']
                    )

                fig.update_layout(hovermode="x unified")
                glm_config['figure'] = fig
                glm_config['last_run_variable'] = selected_var
                glm_config['selected_var'] = selected_var
                glm_config['show_points'] = show_points
                glm_config['color_by'] = color_by

            # Afficher la figure
            if glm_config['figure']:
                st.plotly_chart(glm_config['figure'], use_container_width=True)
                st.session_state.glm_stat['run_triggered'] = False
            
   
            systems_mapping = {"Selected System": selected_predictor}
            with st.spinner('Running GLM models...'):
                glm_results = safe_glm_crossgroup(
                    df_predictors=df_filtered,
                    df_outcomes=df_filtered,
                    outcomes=selected_outcomes,
                    systems=systems_mapping,
                    covariate=selected_covariates,
                    visit_name=title if 'title' in locals() else "GLM Run",
                    interaction_var=selected_interaction if interaction else None,
                    family=family, 
                )
            if glm_results.empty:
                st.error("""
                ❌ No GLM results obtained. Possible causes:
                   Missing data (NaN) in the selected variables
                   Too few observations after cleaning
                   Model convergence issues
                   Inappropriate family/link function for your data
                """)
        
            else:
                st.subheader("📊 GLM Results")
                st.dataframe(glm_results)
                # Visualisation interactive des résultats
                st.subheader("📈 Visualization of results")
                
                # Couleurs pour les systèmes
                base_colors = {
                    'A4B2': '#76b7b2',
                    'M1': '#59a14f',
                    'VAChT': '#edc948',
                    'D1': '#b07aa1',
                    'D2': '#ff9da7',
                    'DAT': '#9c755f',
                    'Nor': '#79706e',
                    '5HT1a': '#86bcb6',
                    '5HT1b': '#d95f02',
                    '5HT2a': '#e7298a',
                    '5HT4': '#66a61e',
                    '5HT6': '#e6ab02',
                    '5HTT': '#a6761d',
                }
           
                prefixes_pre = ['pre_', 'loc_inj_', 'tract_inj_']
                prefixes_post = ['post_', 'loc_inj_', 'tract_inj_']

                keys_pre = ['A4B2', 'M1', 'D1', 'D2', 'Nor', '5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6']
                keys_post = ['VAChT', 'DAT', '5HTT']
                neuro_colors = {}

                for key, color in base_colors.items():
                    neuro_colors[key] = color

                for key in keys_pre:
                    for prefix in prefixes_pre:
                        neuro_colors[prefix + key] = base_colors[key]

                for key in keys_post:
                    for prefix in prefixes_post:
                        neuro_colors[prefix + key] = base_colors[key]
                
                for outcome in glm_results['Outcome'].unique():
                    outcome_data = glm_results[glm_results['Outcome'] == outcome].copy()
                    outcome_data['Clean_Predictor'] = outcome_data['Predictor'].apply(clean_predictor_name)
                    outcome_data = outcome_data[outcome_data['Clean_Predictor'].isin(neuro_colors.keys())]

                    if outcome_data.empty:
                        st.write(f"No relevant predictors to display for outcome {outcome}.")
                        continue

                    fig = go.Figure()

                    for _, row in outcome_data.iterrows():
                        predictor = row['Clean_Predictor']
                        color = neuro_colors.get(predictor, '#1f77b4')

                        fig.add_trace(go.Bar(
                            x=[predictor],
                            y=[row['Coefficient']],
                            name=predictor,
                            marker_color=color,
                            text=[f"p={row['P-value']:.3f}"],
                            textposition='auto',
                            hoverinfo='text',
                            hovertext=(
                                f"Predictor: {predictor}<br>"
                                f"Coefficient: {row['Coefficient']:.3f}<br>"
                                f"p-value: {row['P-value']:.3f}"
                            )
                        ))

                        if row['Significant']:
                            fig.add_annotation(
                                x=predictor,
                                y=row['Coefficient'],
                                text="*",
                                showarrow=False,
                                font=dict(size=20, color='black'),
                                yshift=10
                            )

                    fig.update_layout(
                        title=f"GLM Results for {outcome}",
                        xaxis_title="Predictors",
                        yaxis_title="Coefficient",
                        barmode='group',
                        showlegend=False,
                        hovermode='closest',
                        template='plotly_white'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # ✅ Bouton de téléchargement Excel
                #st.subheader("📥 GLM Results Export")
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    glm_results.to_excel(writer, sheet_name='GLM_Results', index=False)

                st.download_button(
                    label="📥 Download results in Excel format",
                    data=output.getvalue(),
                    file_name=f"glm_results_{selected_system.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


   
    #quand je choisis deux groupes de sujets avec les memes base - pas mme version pour test independant ca marche quand emme ///rajouter une condition pour base-subject
    #sinon l'utilisateur est censé le faire de lui meme ce choix -- modification secondaire
    elif analysis_method == "T-Test":
        st.subheader("T-Test Configuration")
        if 'ttest' not in st.session_state:
            st.session_state.ttest = {
                'ran': False,
                'results_df': None,
                'cleaned_data': None,
                'df1_clean': None,
                'df2_clean': None,
                'variables': None,
                'paired': False,
                'analysis_done': False,
                'data_ready_ttest': False,
                'active_ttest_tab': "Statistic Results",
                'plot_config': {
                    'plot_type': "Violin plot",
                    'selected_var': None,
                    'figure': None,
                    'last_run_variable': None,
                    'last_run_plot_type': None
                    }
            }
           
        # Ajouter le choix du type de test
        paired_test = st.checkbox(
            "Paired data (same subjects in both groups)", 
            value=False,
            key="paired_test"
        )
        
        if analysis_type == "Personalized subject list":
            all_subjects = sorted(df_combined['subject'].unique())
            
            # Groupe 1 
            group1_subjects = st.multiselect(
                "Select group 1 subjects",
                options=all_subjects, 
                key="group1_select"
            )
            #Groupe 2 -- verifier que choix de session possible et juste base du sujet qui revient --no a modifier
            if paired_test:
                # Mode apparié - mêmes sujets
                if group1_subjects:
                    # Extraire les identifiants de base (sans le suffixe avec la version)
                    base_ids = [subj.split('-V')[0] for subj in group1_subjects]
                    available_for_pairing = []
                    for base in base_ids:
                        available_for_pairing.extend([s for s in all_subjects if s.startswith(base)])
                    group2_subjects = st.multiselect(
                        "Select matching subjects for group 2",
                        options=list(set(available_for_pairing)),  # Éviter les doublons
                        default=group1_subjects,
                        key="group2_select_paired"
                    )
                else:
                    group2_subjects = []
            else:
                # Mode indépendant - sujets différents
                available_group2 = [s for s in all_subjects if s not in group1_subjects]
                group2_subjects = st.multiselect(
                    "Select group 2 subjects",
                    options=available_group2,
                    key="group2_select_independent"
                )

        elif analysis_type == "By session and sex":
            if paired_test:
                # Extraire les identifiants de base (sans le suffixe -V1/-V2)
                base_ids_group1 = {subj.split('-V')[0] for subj in group1_subjects}
                base_ids_group2 = {subj.split('-V')[0] for subj in group2_subjects}
                
                # Trouver les bases communes aux deux groupes
                common_base_ids = base_ids_group1 & base_ids_group2
                
                if not common_base_ids:
                    st.error("No matching subject pairs found between the two groups")
                    st.stop()
                
                # Filtrer les sujets pour ne garder que ceux dont la base est commune
                group1_subjects = [subj for subj in group1_subjects 
                                if subj.split('-V')[0] in common_base_ids]
                group2_subjects = [subj for subj in group2_subjects 
                                if subj.split('-V')[0] in common_base_ids]
                
                # Vérifier qu'on a au moins 3 paires
                if len(common_base_ids) < 3:
                    st.warning(f"Only {len(common_base_ids)} paired subjects found (minimum 3 required)")
                
                # Afficher le résultat du matching
                st.success(f"Found {len(common_base_ids)} valid subject pairs for paired analysis")
            elif not paired_test:
                base_ids_group1 = {subj.split('-V')[0] for subj in group1_subjects}
                base_ids_group2 = {subj.split('-V')[0] for subj in group2_subjects}
                overlapping_bases = base_ids_group1 & base_ids_group2

                if overlapping_bases:
                    st.error(f"Independent test requires different subjects – overlapping base IDs found: {', '.join(sorted(overlapping_bases))}")
                    st.stop()
        else:  # Pour By session/sex ou Combined filter
            if paired_test:
                st.warning("Paired test requires explicit subject pairing - use 'Personalized subject list' or 'By session and sex' mode")
                st.stop()
                
        #Vérifications finales
        if not group1_subjects or not group2_subjects:
            st.warning("Please select subjects for both groups")
            st.stop()
            
        if paired_test:
            if len(group1_subjects) != len(group2_subjects):
                st.error("Paired test requires same number of subjects in both groups")
                st.stop()
                
            # Vérifier que ce sont bien les mêmes sujets (versions différentes possibles)
            base_ids_group1 = {subj.split('-V')[0] for subj in group1_subjects}
            base_ids_group2 = {subj.split('-V')[0] for subj in group2_subjects}
            
            if base_ids_group1 != base_ids_group2:
                st.error("Paired test requires matching subject IDs (only version suffix should differ)")
                st.stop()        

        df_group1 = df_combined[df_combined['subject'].isin(group1_subjects)]
        df_group2 = df_combined[df_combined['subject'].isin(group2_subjects)]

        numeric_cols = df_group1.select_dtypes(include=[np.number]).columns.tolist()
        variables_to_compare = st.multiselect(
            "Select variables to compare:",
            options=numeric_cols,
            key="ttest_vars"
        )

        cleaned_data = {}
        if st.button("Run Statistical Comparison"):
            results = []
            
            for var in variables_to_compare:
                if var in df_group1.columns and var in df_group2.columns:
                    # Nettoyage des groupes pour cette variable
                    df1_clean, df2_clean, n_pairs = clean_groups_for_variable(df_group1, df_group2, var, paired_test)
                    cleaned_data[var] = (df1_clean, df2_clean)
                    # Vérification du nombre de sujets après nettoyage et suppression
                    if paired_test:
                        if n_pairs is None or n_pairs < 3:
                            st.warning(f"{var}: Only {n_pairs if n_pairs is not None else 0} valid subject pairs found (minimum 3 required). Skipping.")
                            continue
                        st.markdown(f"**{var}** – Found {n_pairs} valid subject pairs for paired analysis.")
                        all_removed_subjects = [] 
                    else:
                        removed_subjects_g1 = df_group1[df_group1[var].isna()]['subject'].tolist()
                        removed_subjects_g2 = df_group2[df_group2[var].isna()]['subject'].tolist()
                        all_removed_subjects = removed_subjects_g1 + removed_subjects_g2
                        if len(df1_clean) < 3 or len(df2_clean) < 3:
                            st.warning(f"{var}: Not enough valid subjects for independent analysis (≥3 per group). Skipping.") # dire cest quel subject qui a été retiré du à un none
                            if all_removed_subjects:
                                st.markdown(f"<span style='color:grey'>Removed due to missing values: {', '.join(all_removed_subjects)}</span>", unsafe_allow_html=True)
                                st.markdown(f"**{var}** – Number of valid subjects: Group 1 = {len(df1_clean)}, Group 2 = {len(df2_clean)}")
                            continue

                    if all_removed_subjects:
                        st.markdown(f"<span style='color:grey'>Removed due to missing values: {', '.join(all_removed_subjects)}</span>", unsafe_allow_html=True)


                    test_results = perform_group_comparison(
                        df1_clean[var],
                        df2_clean[var],
                        paired=paired_test  
                    )
                    if test_results:
                        test_results['variable'] = var  
                        results.append(test_results)

            
            if not results:
                st.error("No valid comparisons could be performed")
                st.stop()
                
            results_df = pd.DataFrame(results).reset_index(drop=True)
            # Supprime les colonnes dupliquées
            results_df = results_df.loc[:, ~results_df.columns.duplicated()]
            
            # Réorganisation des colonnes 
            cols_order = [
                'variable', 'test_type', 'p_value', 'significant',
                'mean_group1', 'mean_group2', 'effect_size', 'power',
                'n_group1', 'n_group2', 'statistic',
                'shapiro_p1', 'shapiro_p2'
            ]
            # Ajoutez 'levene_p' seulement si présent (sujets indépendants)
            if 'levene_p' in results_df.columns:
                cols_order.append('levene_p')
            cols_order = [c for c in cols_order if c in results_df.columns]

            # Stockage des résultats
            st.session_state.ttest.update({
                'results_df': results_df,
                'cleaned_data': cleaned_data,
                'df1_clean': df1_clean,
                'df2_clean': df2_clean,
                'variables': variables_to_compare,
                'paired': paired_test, 
                'analysis_done' : True,
                'data_ready_ttest': True,
                'ran': True,
            })


    
        # ---------  Affichage ------------
        if st.session_state.ttest.get('ran', False) and st.session_state.ttest['data_ready_ttest']:
            config = st.session_state.ttest['plot_config']
            tab_options = ["Statistic Results", "Visualization"]
            selected_tab = st.radio(
                "Navigation",
                tab_options,
                horizontal=True,
                index=tab_options.index(st.session_state.ttest['active_ttest_tab']),
                key='tab_selector_ttest',
                label_visibility="hidden"
            )
            st.session_state.ttest['active_ttest_tab'] = selected_tab

            #---------- Onglet 1 : Resultats T-Test ----------
            if selected_tab == "Statistic Results":
                st.subheader("T-Test Results")

                format_dict = {
                    'p_value': '{:.4f}',
                    'effect_size': '{:.3f}',
                    'power': '{:.3f}',
                    'mean_group1': '{:.3f}',
                    'mean_group2': '{:.3f}',
                    'shapiro_p1': '{:.4f}',
                    'shapiro_p2': '{:.4f}'
                }
                if 'levene_p' in st.session_state.ttest['results_df'].columns:
                    format_dict['levene_p'] = '{:.4f}'

                if st.session_state.ttest['results_df'].columns.is_unique and st.session_state.ttest['results_df'].index.is_unique:
                    st.dataframe(st.session_state.ttest['results_df'].style.format(format_dict).applymap(
                        lambda x: 'background-color: yellow' if isinstance(x, float) and x < 0.05 else '',
                        subset=[col for col in ['p_value', 'shapiro_p1', 'shapiro_p2', 'levene_p'] if col in st.session_state.ttest['results_df'].columns]
                    ))
                else:
                    st.warning("Colonnes ou index non uniques : affichage sans style.")
                    st.dataframe(st.session_state.ttest['results_df'])

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    st.session_state.ttest['results_df'].to_excel(writer, index=False, sheet_name='T-Test Results')

                st.download_button(
                    label="Download results as Excel",
                    data=output.getvalue(),
                    file_name="statistical_results.xlsx",
                    mime="applicat"
                    "ion/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            #---------- Onglet 2 : Visualisation ----------
            elif selected_tab == "Visualization":
        
                st.subheader("Group Comparison Plots")

                plot_type = st.radio(
                    "Choose plot type:",
                    ["Violin plot", "Box plot"],
                    index=["Violin plot", "Box plot"].index(config['plot_type']),
                    key="plot_type_radio"
                )

                selected_var = st.selectbox(
                    "Select variable to display:",
                    options=st.session_state.ttest['variables'],
                    index=st.session_state.ttest['variables'].index(config['selected_var']) if config['selected_var'] in st.session_state.ttest['variables'] else 0,
                    key="selected_var_box"
                )

                config['plot_type'] = plot_type
                config['selected_var'] = selected_var

                if (selected_var != config.get('last_run_variable')) or (plot_type != config.get('last_run_plot_type')):

                    if selected_var in st.session_state.ttest['cleaned_data']:
                        df1_clean, df2_clean = st.session_state.ttest['cleaned_data'][selected_var]

                        if not df1_clean.empty and not df2_clean.empty:
                            df_plot = pd.concat([
                                df1_clean[[selected_var]].assign(Group="Group 1"),
                                df2_clean[[selected_var]].assign(Group="Group 2")
                            ])

                            if plot_type == "Violin plot":
                                fig = go.Figure()
                                colors = ['#1f77b4', '#ff7f0e']
                                for i, group in enumerate(df_plot["Group"].unique()):
                                    fig.add_trace(go.Violin(
                                        y=df_plot[df_plot["Group"] == group][selected_var],
                                        name=group,
                                        box_visible=True,
                                        meanline_visible=True,
                                        points="all",
                                        line_color='black',
                                        fillcolor=colors[i]
                                    ))
                            else:
                                fig = px.box(
                                    df_plot,
                                    x="Group",
                                    y=selected_var,
                                    points="all",
                                    color="Group",
                                    color_discrete_sequence=['#1f77b4', '#ff7f0e']
                                )

                            fig.update_layout(
                                title=f"{selected_var} - Group Comparison",
                                height=500,
                                margin=dict(l=20, r=20, t=60, b=20),
                                showlegend=True
                            )

                            config['figure'] = fig
                            config['last_run_variable'] = selected_var
                            config['last_run_plot_type'] = plot_type
                        else:
                            st.warning("Not enough data to generate plot.")
                    else:
                        st.warning(f"No cleaned data found for variable: {selected_var}")

                if config['figure']:
                    st.plotly_chart(config['figure'], use_container_width=True)

           

    elif analysis_method =="Correlation":
        st.subheader("🔗 Correlation Analysis")

        if 'corr' not in st.session_state:
            st.session_state.corr = {
                'ran': False,
                'show_all': False,
                'p_thresh': 0.05,
                'active_tab': "Visualization",
                'data_ready': False,
                'data': {
                    'corr_matrix': None,
                    'pval_matrix': None,
                    'cross_corr': None, 
                    'cross_pvals': None,
                    'session1': "",
                    'session2': ""
                    }
            }

        # === Configuration des variables ===
        st.markdown("### Correlation Variables Configuration")
        df_filtered = df_combined[df_combined['subject'].isin(group1_subjects + group2_subjects)]

        # Types de variables disponibles
        system_options = {
            "Synaptic ratio": [
                "pre_A4B2", "pre_M1", "pre_D1", "pre_D2",
                "pre_5HT1a", "pre_5HT1b", "pre_5HT2a",
                "pre_5HT4", "pre_5HT6",
                "post_VAChT", "post_DAT", "post_5HTT"
            ],
            "Neurotransmitter (Loc)": [f"loc_inj_{sys}" for sys in [
                "GABAa", "mGluR5", "MU", "H3", "CB1", "A4B2", "M1", "VAChT",
                "D1", "D2", "DAT", "Nor", "5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"
            ] if f"loc_inj_{sys}" in df_filtered.columns],

            "Neurotransmitter (Tract)": [f"tract_inj_{sys}" for sys in [
                "GABAa", "mGluR5", "MU", "H3", "CB1", "A4B2", "M1", "VAChT",
                "D1", "D2", "DAT", "Nor", "5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"
            ] if f"tract_inj_{sys}" in df_filtered.columns],

            "Clinical Outcomes": [col for col in df_filtered.columns 
                        if col not in ['subject', 'Sexe_bin', 'sex', 'lesion_volume']
                        and not col.startswith(('loc_inj_', 'tract_inj_', 'pre_', 'post_'))]
        } 

        with col1:
            system_type1 = st.radio(
                "Variable type for Set 1:",
                options=list(system_options.keys()),
                key="system_type1"
            )
            vars1 = system_options[system_type1]
            
        with col2:
            system_type2 = st.radio(
                "Variable type for Set 2:",
                options=list(system_options.keys()),
                key="system_type2"
            )
            vars2 = system_options[system_type2]
      
        # === Exécution ===
        if st.button("🚀 Run Correlation Analysis", key="run_corr_button"):
            if not group1_subjects or not group2_subjects:
                st.error("Please select subjects for both groups")
                st.stop()

            if not vars1 or not vars2:
                st.error("Please select variables for both sets")
                st.stop()

            try:
                df_filtered["subject_base"] = df_filtered["subject"].apply(extract_subject_id)

                if session1 != "combined":
                    df1 = df_filtered[
                        (df_filtered["subject"].isin(group1_subjects)) &
                        (df_filtered["subject"].str.contains(f"_ses-{session1}"))
                    ][["subject_base"] + vars1].drop_duplicates()
                else:
                    df1 = df_filtered[
                        df_filtered["subject"].isin(group1_subjects)
                    ][["subject_base"] + vars1].drop_duplicates()

                if session2 != "combined":
                    df2 = df_filtered[
                        (df_filtered["subject"].isin(group2_subjects)) &
                        (df_filtered["subject"].str.contains(f"_ses-{session2}"))
                    ][["subject_base"] + vars2].drop_duplicates()
                else:
                    df2 = df_filtered[
                        df_filtered["subject"].isin(group2_subjects)
                    ][["subject_base"] + vars2].drop_duplicates()

                if session1 == "combined"or session2 == "combined":
                    suffix1 = "_set1"
                    suffix2 = "_set2"
                else:
                    suffix1 = f"_{session1}_1" if session1 == session2 else f"_{session1}"
                    suffix2 = f"_{session2}_2" if session1 == session2 else f"_{session2}"

                df1_renamed = df1.rename(columns={col: col + suffix1 for col in vars1})
                df2_renamed = df2.rename(columns={col: col + suffix2 for col in vars2})
                
                # Vérification des sujets communs
                common_ids = sorted(set(df1_renamed["subject_base"]) & set(df2_renamed["subject_base"]))
                if not len(common_ids) > 3:
                    st.error(f"Only {len(common_ids)} common subjects found (minimum 3 required)")
                    st.stop()
                
                # Fusion finale pour avoir un dataframe pour la matrice de correlation
                df_corr = df1_renamed[df1_renamed["subject_base"].isin(common_ids)].merge(
                    df2_renamed[df2_renamed["subject_base"].isin(common_ids)],
                    on="subject_base"
                )

                st.write(f"Final dataset shape: {df_corr.shape}")
                if df_corr.shape[0] < 3:
                    st.error("Not enough valid subjects after merging datasets")
                    st.stop()

                # Calcul des corrélations
                with st.spinner('Calculating correlations...'):
                    corr_matrix, pval_matrix = get_correlation_matrix(df_corr)

                    set1_cols = [col for col in corr_matrix.columns if col.endswith(suffix1)]
                    set2_cols = [col for col in corr_matrix.columns if col.endswith(suffix2)]
                    set1_cols_p=[col for col in pval_matrix.columns if col.endswith(suffix1)]
                    set2_cols_p = [col for col in pval_matrix.columns if col.endswith(suffix2)]
                    cross_corr = corr_matrix.loc[set1_cols, set2_cols]
                    cross_pvals = pval_matrix.loc[set1_cols_p, set2_cols_p]

            
                # préparation pour affichage
                st.success("Analysis completed!")
        
                st.session_state.corr.update({
                    'ran': True,
                    'data_ready': True,
                    'data': {
                        'corr_matrix': corr_matrix,
                        'pval_matrix': pval_matrix,
                        'cross_corr': cross_corr,
                        'cross_pvals': cross_pvals,
                        'session1': session1,
                        'session2': session2
                    }
                })
            except Exception as e:
                st.error(f"Error during correlation calculation: {str(e)}")
                st.session_state.corr['ran'] = False

        # ---------  Affichage ------------
        if st.session_state.corr.get('ran', False) and st.session_state.corr['data_ready']:
            data = st.session_state.corr['data']
            tab_names = ["Full Matrix", "Cross Matrix", "Visualization"]
            selected_tab = st.radio(
                "Navigation",
                tab_names,
                horizontal=True,
                index=tab_names.index(st.session_state.corr['active_tab']),
                key='tab_selector',
                label_visibility="hidden"
            )
            st.session_state.corr['active_tab'] = selected_tab

            # Full Matrix Tab
            if selected_tab == "Full Matrix":
                st.write("### Full Correlation Matrix")
                st.dataframe(st.session_state.corr['data']['corr_matrix'].style.format("{:.2f}"))
                st.write("### P-Value Matrix")
                st.dataframe(st.session_state.corr['data']['pval_matrix'].style.format("{:.4f}"))

            # Cross Matrix Tab
            elif selected_tab == "Cross Matrix":
                st.write(f"### {st.session_state.corr['data']['session1']} vs {st.session_state.corr['data']['session2']} Correlations")
                st.dataframe(st.session_state.corr['data']['cross_corr'].style.format("{:.2f}"))
                st.write("### Significant Correlations")
                sig_corrs = st.session_state.corr['data']['cross_corr'].where(
                    st.session_state.corr['data']['cross_pvals'] < st.session_state.corr['p_thresh']
                )
                st.dataframe(sig_corrs.style.format("{:.2f}"))

            #  Visualization Tab avec heatmap Plotly
            elif selected_tab == "Visualization":
                col1, col2 = st.columns(2)
                with col1:
                    show_all = st.checkbox(
                        "Show all correlations (incl. non-significant)",
                        value=st.session_state.corr.get('show_all', False),
                        key='corr_show_all'
                    )
                    st.session_state.corr['show_all'] = show_all
                with col2:
                    # p_thresh = st.slider(
                    #     "p-value threshold",
                    #     0.001, 0.1,
                    #     value=st.session_state.corr.get('p_thresh', 0.05),
                    #     step=0.001,
                    #     key='corr_p_thresh'
                    # )
                    p_values = [round(x, 3) for x in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                                  0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]]

                    p_thresh = st.selectbox(
                        "p-value threshold",
                        options=p_values,
                        index=p_values.index(st.session_state.corr.get('p_thresh', 0.05)),
                        key='corr_p_thresh'
                    )
                    st.session_state.corr['p_thresh'] = p_thresh

                # Préparer la matrice à afficher : masquer si nécessaire les corrélations non-significatives selon le seuil choisit par l'utilisateur
                cross_corr_plot = data['cross_corr'].copy()
                corr = data['cross_corr'].values
                pvals = data['cross_pvals'].values
                x = data['cross_corr'].columns.tolist()
                y = data['cross_corr'].index.tolist()

                # if not show_all:
                #     mask = data['cross_pvals'] >= p_thresh
                #     cross_corr_plot = cross_corr_plot.mask(mask)
                mask = ((data['cross_pvals'] >= p_thresh) & (~show_all)).values 

                #superposition de deux matrices pour griser les cases non significatives
                fig = go.Figure(go.Heatmap(
                    z=corr,
                    x=list(range(len(x))),
                    y=list(range(len(y))),
                    colorscale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    text=np.round(corr, 2),
                    texttemplate="%{text}",
                    hoverongaps=False,
                    colorbar=dict(
                        title=dict(
                            text="Correlation",
                            font=dict(color="black")
                        ),
                        tickfont=dict(color="black")
                    )
                ))

                shapes = []
                for i in range(len(y)):
                    for j in range(len(x)):
                        if mask[i, j]:
                            shapes.append(dict(
                                type="rect",
                                xref="x",
                                yref="y",
                                x0=j - 0.5,
                                x1=j + 0.5,
                                y0=i - 0.5,
                                y1=i + 0.5,
                                fillcolor="rgba(200,200,200,0.5)",
                                line_width=0,
                                layer="above"
                            ))
                fig.update_traces(
                    hoverongaps=False
                )

                fig.update_layout(
                    shapes=shapes,
                    margin=dict(l=40, r=40, t=40, b=40),
                    autosize=True,
                    height=600,  
                    width=800, 
                    yaxis=dict(
                        #automargin=True,
                        scaleanchor="x",
                        scaleratio=1,
                        autorange='reversed',
                        tickvals=list(range(len(y))),
                        ticktext=y,
                        ticks="outside",
                        ticklen=5,
                        tickwidth=1,
                        tickcolor='#000',
                        showgrid=False,
                        tickfont=dict(color='black'),
                        zeroline=False
                    ),
                    xaxis=dict(
                        tickvals=list(range(len(x))),
                        ticktext=x,
                        tickangle=45,
                        ticks="outside",
                        ticklen=5,
                        tickwidth=1,
                        tickcolor='#000',
                        showgrid=False,
                        tickfont=dict(color='black'),
                        zeroline=False

                    ),
                )

                st.plotly_chart(fig, use_container_width=True)
    
                # Plotly heatmap mais avec effacement des valeurs non-significative
                # fig = px.imshow(
                #     cross_corr_plot,
                #     color_continuous_scale='RdBu_r',
                #     zmin=-1, zmax=1,
                #     text_auto=".2f",
                #     aspect="equal",
                #     labels=dict(color="Correlation")
                # )
                # fig.update_traces(
                #     hoverongaps=False
                # )
                # fig.update_layout(
                #     title=f"Correlations {data['session1']} vs {data['session2']}",
                #     margin=dict(l=40, r=40, t=40, b=40),
                #     coloraxis_colorbar=dict(title="Correlation"),
                #     xaxis=dict(tickangle=45, showgrid=False, zeroline=False,tickfont=dict(color='black')),
                #     yaxis=dict(showgrid=False, zeroline=False,tickfont=dict(color='black') ),
              
                # )
                
                # st.plotly_chart(fig, use_container_width=True)
                
                # Exporter en format PDF (format csv déjà disponible sur la figure interactive pour chaque table séparément)
                if st.session_state.corr['data_ready']:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        st.session_state.corr['data']['corr_matrix'].to_excel(writer, sheet_name='Full_Correlation')
                        st.session_state.corr['data']['pval_matrix'].to_excel(writer, sheet_name='Full_Pvalues')
                        st.session_state.corr['data']['cross_corr'].to_excel(writer, sheet_name='Cross_Correlation')
                        st.session_state.corr['data']['cross_pvals'].to_excel(writer, sheet_name='Cross_Pvalues')
                    
                    st.download_button(
                        label="📥 Download the results",
                        data=output.getvalue(),
                        file_name=f"correlation_{st.session_state.corr['data']['session1']}_vs_{st.session_state.corr['data']['session2']}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                

