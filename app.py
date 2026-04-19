import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import io, base64, os, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesIQ",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  DARK-MODE CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d !important;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
textarea {
    background-color: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
[data-testid="stSlider"] { accent-color: #00d4aa; }
[data-testid="stCheckbox"] { accent-color: #00d4aa; }
.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #0ea5e9) !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-size: 1rem !important;
    transition: opacity .2s, transform .1s !important;
}
.stButton > button:hover { opacity: .88; transform: translateY(-1px); }
.page-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4aa, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 4px;
    letter-spacing: -1px;
}
.page-subtitle {
    text-align: center;
    color: #8b949e;
    font-size: 1rem;
    margin-bottom: 32px;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #00d4aa;
    margin: 28px 0 12px 0;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
}
.glass-card {
    background: rgba(22,27,34,0.85);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 16px;
}
.metric-card {
    background: linear-gradient(135deg, #161b22, #21262d);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    transition: border-color .2s;
}
.metric-card:hover { border-color: #00d4aa; }
.metric-value { font-size: 2.2rem; font-weight: 800; color: #00d4aa; line-height: 1; }
.metric-label { font-size: 0.78rem; color: #8b949e; margin-top: 4px;
                text-transform: uppercase; letter-spacing: 1px; }
.acc-badge { display:inline-block; padding:4px 14px; border-radius:20px;
             font-weight:700; font-size:1rem; }
.risk-low    { background:#00d4aa22; color:#00d4aa; border:1px solid #00d4aa; }
.risk-medium { background:#f59e0b22; color:#f59e0b; border:1px solid #f59e0b; }
.risk-high   { background:#ef444422; color:#ef4444; border:1px solid #ef4444; }
.footer {
    text-align:center; color:#484f58; padding:32px 0 16px;
    font-size:.78rem; border-top:1px solid #21262d; margin-top:48px;
}
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PLOT PALETTE
# ─────────────────────────────────────────────
DARK_BG = "#0d1117"
CARD_BG = "#161b22"
BORDER  = "#30363d"
ACCENT  = "#00d4aa"
ACCENT2 = "#0ea5e9"
TEXT    = "#e6edf3"
MUTED   = "#8b949e"

def apply_dark(fig, axes=None):
    fig.patch.set_facecolor(CARD_BG)
    for ax in (axes or fig.axes):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=MUTED)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for s in ax.spines.values():
            s.set_edgecolor(BORDER)

# ─────────────────────────────────────────────
#  KAGGLE DOWNLOAD
# ─────────────────────────────────────────────
DATASET_SLUG = "hossamhassan1/diabetes-classification"
CSV_PATH     = "diabetes.csv"

def download_from_kaggle(username=None, key=None):
    try:
        if username and key:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"]      = key
        import kaggle
        kaggle.api.authenticate()
        with st.spinner("⬇️ Downloading dataset from Kaggle…"):
            kaggle.api.dataset_download_files(DATASET_SLUG, path=".", unzip=True)
        if not os.path.exists(CSV_PATH):
            for root, _, files in os.walk("."):
                for f in files:
                    if f.lower() == "diabetes.csv":
                        os.rename(os.path.join(root, f), CSV_PATH); break
        return os.path.exists(CSV_PATH)
    except Exception as e:
        st.error(f"Kaggle download failed: {e}"); return False

@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return None

def ensure_dataset():
    if os.path.exists(CSV_PATH): return True
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    has_creds   = os.path.exists(kaggle_json) or (
        "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ)
    if has_creds:
        if download_from_kaggle():
            st.cache_data.clear(); return True
    st.warning("⚠️ `diabetes.csv` not found. Enter Kaggle credentials below.")
    st.markdown("Get token: [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**")
    with st.form("kaggle_creds"):
        u = st.text_input("Kaggle Username")
        k = st.text_input("Kaggle API Key", type="password")
        if st.form_submit_button("Download Dataset"):
            if not u or not k: st.error("Both fields required.")
            elif download_from_kaggle(u, k):
                st.success("✅ Downloaded!"); st.cache_data.clear(); st.rerun()
    st.stop()

# ─────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_data(data):
    df = data.copy()
    zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    for col in ['Glucose','BloodPressure']:
        df[col] = df[col].fillna(df[col].mean())
    for col in ['SkinThickness','Insulin','BMI']:
        df[col] = df[col].fillna(df[col].median())
    # IQR outlier capping
    for col in [c for c in df.columns if c != 'Outcome']:
        Q1, Q3 = df[col].quantile(.25), df[col].quantile(.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    return df

def engineer_features(df):
    df = df.copy()
    df['Glucose_BMI']     = df['Glucose'] * df['BMI']
    df['Age_Glucose']     = df['Age']     * df['Glucose']
    df['Insulin_Glucose'] = df['Insulin'] * df['Glucose']
    df['BMI_Age']         = df['BMI']     * df['Age']
    return df

def create_features_target(df, use_fe=True):
    if use_fe: df = engineer_features(df)
    sc = StandardScaler()
    X  = pd.DataFrame(sc.fit_transform(df.drop('Outcome', axis=1)),
                      columns=df.drop('Outcome', axis=1).columns)
    y  = df['Outcome']
    return X, y, sc

# ─────────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────────
def build_models(rs=42):
    rf  = RandomForestClassifier(n_estimators=300, max_depth=8,
            min_samples_split=4, min_samples_leaf=2,
            max_features='sqrt', class_weight='balanced', random_state=rs)
    xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=5,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=2, eval_metric='logloss', verbosity=0, random_state=rs)
    gb  = GradientBoostingClassifier(n_estimators=200, max_depth=4,
            learning_rate=0.05, subsample=0.8, random_state=rs)
    lr  = LogisticRegression(C=0.5, max_iter=2000, solver='lbfgs',
            class_weight='balanced', random_state=rs)
    knn = KNeighborsClassifier(n_neighbors=11, weights='distance')
    ens = VotingClassifier(
            estimators=[('rf',rf),('xgb',xgb_m),('gb',gb),('lr',lr)],
            voting='soft', weights=[2,2,1,1])
    return {"Random Forest": rf, "XGBoost": xgb_m,
            "Gradient Boosting": gb, "Logistic Regression": lr,
            "KNN": knn, "🏆 Ensemble": ens}

def get_report_df(y_true, y_pred):
    return pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True)).T.round(3)

# ─────────────────────────────────────────────
#  DARK PLOT HELPERS
# ─────────────────────────────────────────────
def plot_confusion(y_test, y_pred, name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    apply_dark(fig)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("t",[DARK_BG, ACCENT])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                linewidths=1, linecolor=BORDER,
                xticklabels=['No DM','DM'], yticklabels=['No DM','DM'],
                annot_kws={"size":16,"weight":"bold","color":TEXT})
    ax.set_title(name, color=TEXT, pad=10)
    ax.set_xlabel("Predicted", color=MUTED)
    ax.set_ylabel("Actual", color=MUTED)
    plt.tight_layout(); return fig

def plot_roc(models_dict, y_test, X_test):
    fig, ax = plt.subplots(figsize=(7,5))
    apply_dark(fig)
    colors = [ACCENT, ACCENT2, "#f59e0b", "#a78bfa", "#f472b6", "#fb923c"]
    ax.plot([0,1],[0,1],'--',color=BORDER,lw=1)
    for (name, model), color in zip(models_dict.items(), colors):
        prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        score = auc(fpr, tpr)
        short = name.replace("🏆 ","")
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{short}  AUC={score:.3f}")
    ax.set_xlabel("False Positive Rate", color=MUTED)
    ax.set_ylabel("True Positive Rate", color=MUTED)
    ax.set_title("ROC Curves — All Models", color=TEXT, fontsize=13)
    ax.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    plt.tight_layout(); return fig

def plot_importance(model, feature_names, title):
    if hasattr(model,'feature_importances_'):   imp = model.feature_importances_
    elif hasattr(model,'coef_'):                imp = np.abs(model.coef_[0])
    else:                                        return None
    df = pd.DataFrame({'Feature':feature_names,'Importance':imp}).sort_values('Importance')
    fig, ax = plt.subplots(figsize=(7, max(4, len(feature_names)*0.42)))
    apply_dark(fig)
    norm = plt.Normalize(df['Importance'].min(), df['Importance'].max())
    bars = ax.barh(df['Feature'], df['Importance'], edgecolor=BORDER)
    for bar, val in zip(bars, df['Importance']):
        bar.set_facecolor(plt.cm.cool(norm(val)))
    ax.set_title(title, color=TEXT, fontsize=12)
    ax.set_xlabel("Importance", color=MUTED)
    plt.tight_layout(); return fig

def plot_hist(data, col):
    fig, ax = plt.subplots(figsize=(6,3.5))
    apply_dark(fig)
    for outcome, color, label in [(0,ACCENT2,'No Diabetes'),(1,ACCENT,'Diabetes')]:
        ax.hist(data[data['Outcome']==outcome][col].dropna(),
                bins=25, alpha=0.6, color=color, label=label, edgecolor='none')
    ax.set_xlabel(col,color=MUTED); ax.set_ylabel("Count",color=MUTED)
    ax.set_title(f"Distribution — {col}",color=TEXT)
    ax.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT)
    plt.tight_layout(); return fig

def plot_corr(data):
    fig, ax = plt.subplots(figsize=(10,8))
    apply_dark(fig)
    sns.heatmap(data.corr(), annot=True, fmt='.2f', ax=ax,
                cmap=sns.diverging_palette(180,20,as_cmap=True),
                linewidths=.5, linecolor=BORDER,
                annot_kws={"size":9,"color":TEXT})
    ax.set_title("Correlation Matrix", color=TEXT, fontsize=13)
    plt.tight_layout(); return fig

# ─────────────────────────────────────────────
#  COMPONENT HELPERS
# ─────────────────────────────────────────────
def metric_card(value, label, suffix=""):
    return f"""<div class='metric-card'>
        <div class='metric-value'>{value}{suffix}</div>
        <div class='metric-label'>{label}</div></div>"""

def acc_badge(acc):
    cls = "risk-low" if acc>=85 else "risk-medium" if acc>=75 else "risk-high"
    return f'<span class="acc-badge {cls}">{acc:.1f}%</span>'

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
        <span style='font-size:2.5rem'>🩺</span>
        <div style='font-size:1.3rem;font-weight:800;
             background:linear-gradient(135deg,#00d4aa,#0ea5e9);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent'>DiabetesIQ</div>
        <div style='font-size:.75rem;color:#8b949e'>ML Classification Suite</div>
    </div>
    <hr style='border-color:#30363d;margin:12px 0'>
    """, unsafe_allow_html=True)

    page = st.radio("", ["🏠  Home","🔬  Data Explorer","🤖  Model Lab","🎯  Predict"])

    if 'accuracies' in st.session_state:
        st.markdown("<hr style='border-color:#30363d'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:.75rem;color:#8b949e;text-transform:uppercase;"
                    "letter-spacing:1px;margin-bottom:8px'>Model Scores</div>",
                    unsafe_allow_html=True)
        for name, acc in st.session_state['accuracies'].items():
            short = name.replace("🏆 ","")
            st.markdown(f"""
            <div style='margin:6px 0'>
              <div style='display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:3px'>
                <span style='color:#e6edf3'>{short}</span>
                <span style='color:#00d4aa;font-weight:700'>{acc:.1f}%</span>
              </div>
              <div style='background:#21262d;border-radius:4px;height:5px'>
                <div style='width:{min(acc,100):.0f}%;background:linear-gradient(90deg,#00d4aa,#0ea5e9);
                            height:5px;border-radius:4px'></div>
              </div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HOME PAGE
# ─────────────────────────────────────────────
def home_page():
    st.markdown("<div class='page-title'>DiabetesIQ</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Machine Learning Classification Suite · Pima Indians Diabetes Dataset</div>",
                unsafe_allow_html=True)

    data = load_data()
    if data is None: return
    proc = preprocess_data(data)
    n_pos = int(data['Outcome'].sum())
    n_neg = len(data) - n_pos

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(metric_card(len(data), "Total Records"), unsafe_allow_html=True)
    c2.markdown(metric_card(data.shape[1]-1, "Features"), unsafe_allow_html=True)
    c3.markdown(metric_card(n_pos, "Diabetic Cases"), unsafe_allow_html=True)
    c4.markdown(metric_card(f"{n_pos/len(data)*100:.1f}", "Prevalence", "%"), unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📋 Dataset Preview</div>", unsafe_allow_html=True)
    st.dataframe(data.head(10), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-title'>📊 Class Balance</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,3.5))
        apply_dark(fig)
        bars = ax.bar(['No Diabetes','Diabetes'], [n_neg, n_pos],
                      color=[ACCENT2, ACCENT], edgecolor=BORDER, width=0.5)
        for bar, val in zip(bars, [n_neg, n_pos]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                    str(val), ha='center', va='bottom', color=TEXT,
                    fontsize=11, fontweight='bold')
        ax.set_ylabel("Count", color=MUTED)
        plt.tight_layout()
        st.pyplot(fig); plt.clf()
    with col_b:
        st.markdown("<div class='section-title'>📈 Feature Statistics</div>", unsafe_allow_html=True)
        st.dataframe(proc.describe().T.round(2), use_container_width=True)

    st.markdown("<div class='section-title'>💾 Export</div>", unsafe_allow_html=True)
    if st.button("⬇️  Download Cleaned Dataset"):
        b64 = base64.b64encode(proc.to_csv(index=False).encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="clean_diabetes.csv" '
                    f'style="color:#00d4aa;font-weight:700">Click to download clean_diabetes.csv</a>',
                    unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card' style='margin-top:24px'>
        <b style='color:#00d4aa'>ℹ️ How to use</b><br><br>
        <b>1.</b> <span style='color:#8b949e'>Explore patterns in</span>
        <b style='color:#0ea5e9'>🔬 Data Explorer</b><br>
        <b>2.</b> <span style='color:#8b949e'>Train 6 models in</span>
        <b style='color:#0ea5e9'>🤖 Model Lab</b><br>
        <b>3.</b> <span style='color:#8b949e'>Enter patient data in</span>
        <b style='color:#0ea5e9'>🎯 Predict</b>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DATA EXPLORER
# ─────────────────────────────────────────────
def data_exploration_page():
    st.markdown("<div class='page-title'>🔬 Data Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Understand the data before training</div>", unsafe_allow_html=True)

    data = load_data()
    if data is None: return
    proc = preprocess_data(data)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.checkbox("Dataset info"):
            buf = io.StringIO(); data.info(buf=buf)
            st.code(buf.getvalue())
    with col_b:
        if st.checkbox("Descriptive statistics"):
            st.dataframe(data.describe().round(2), use_container_width=True)

    st.markdown("<div class='section-title'>📊 Feature Distribution</div>", unsafe_allow_html=True)
    features = [c for c in data.columns if c != 'Outcome']
    col = st.selectbox("Select feature", features)

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_hist(proc, col)); plt.clf()
    with c2:
        fig, ax = plt.subplots(figsize=(6,3.5))
        apply_dark(fig)
        d0 = proc[proc['Outcome']==0][col].dropna()
        d1 = proc[proc['Outcome']==1][col].dropna()
        bp = ax.boxplot([d0, d1], patch_artist=True,
                        labels=['No Diabetes','Diabetes'],
                        boxprops=dict(color=ACCENT),
                        whiskerprops=dict(color=MUTED),
                        capprops=dict(color=MUTED),
                        medianprops=dict(color=ACCENT, linewidth=2),
                        flierprops=dict(marker='o', color=ACCENT2, markersize=4))
        bp['boxes'][0].set_facecolor(ACCENT2+"44")
        bp['boxes'][1].set_facecolor(ACCENT +"44")
        ax.set_title(f"Box Plot — {col}", color=TEXT)
        ax.set_ylabel(col, color=MUTED)
        plt.tight_layout()
        st.pyplot(fig); plt.clf()

    st.markdown("<div class='section-title'>🔥 Correlation Heatmap</div>", unsafe_allow_html=True)
    if st.checkbox("Show heatmap"):
        st.pyplot(plot_corr(proc)); plt.clf()

    st.markdown("<div class='section-title'>🔷 Pairplot</div>", unsafe_allow_html=True)
    if st.checkbox("Show pairplot (~10 s)"):
        with st.spinner("Rendering…"):
            sns.set_theme(style="dark", rc={
                "axes.facecolor": DARK_BG, "figure.facecolor": CARD_BG,
                "text.color": TEXT, "axes.labelcolor": MUTED,
                "xtick.color": MUTED, "ytick.color": MUTED})
            fig = sns.pairplot(proc, hue="Outcome",
                               palette={0:ACCENT2, 1:ACCENT},
                               diag_kind="kde",
                               plot_kws={"alpha":.5,"s":15}, height=2.2)
            fig.figure.patch.set_facecolor(CARD_BG)
            st.pyplot(fig); plt.clf()
        sns.reset_defaults()

# ─────────────────────────────────────────────
#  MODEL LAB
# ─────────────────────────────────────────────
def model_training_page():
    st.markdown("<div class='page-title'>🤖 Model Lab</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Train, compare and evaluate 6 classifiers</div>", unsafe_allow_html=True)

    data = load_data()
    if data is None: return
    proc = preprocess_data(data)

    st.markdown("<div class='section-title'>⚙️ Training Settings</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        test_size = st.slider("Test split", 0.10, 0.40, 0.20, 0.05,
                              help="Smaller = more training data = better accuracy")
    with c2:
        rs = st.number_input("Random seed", 0, 999, 42)
    with c3:
        use_fe = st.checkbox("Feature engineering", value=True,
                             help="Adds Glucose×BMI, Age×Glucose, etc. — big accuracy boost")
    use_smote = st.checkbox("SMOTE oversampling", value=True,
                            help="Balances class distribution in training set")

    X, y, scaler = create_features_target(proc, use_fe)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rs, stratify=y)

    if use_smote:
        X_train, y_train = SMOTE(random_state=rs).fit_resample(X_train, y_train)
        st.success(f"SMOTE applied — {pd.Series(y_train).value_counts().to_dict()}")

    if st.button("🚀  Train All Models", use_container_width=True):
        models    = build_models(rs)
        results   = {}
        accs      = {}
        cv_scores = {}
        skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)

        bar = st.progress(0, text="Starting…")
        for i, (name, model) in enumerate(models.items()):
            bar.progress(i/len(models), text=f"Training {name}…")
            model.fit(X_train, y_train)
            preds         = model.predict(X_test)
            acc           = model.score(X_test, y_test) * 100
            cv            = cross_val_score(model, X, y, cv=skf, scoring='accuracy') * 100
            results[name] = (model, preds, acc)
            accs[name]    = acc
            cv_scores[name] = cv
        bar.progress(1.0, text="✅ All models trained!")

        st.session_state['trained_models'] = {n: r[0] for n, r in results.items()}
        st.session_state['accuracies']     = accs
        st.session_state['scaler']         = scaler
        st.session_state['feature_names']  = list(X.columns)
        st.session_state['use_fe']         = use_fe

        # ── Leaderboard ──
        st.markdown("<div class='section-title'>🏆 Accuracy Leaderboard</div>", unsafe_allow_html=True)
        sorted_accs = sorted(accs.items(), key=lambda x: x[1], reverse=True)
        cols = st.columns(len(models))
        for col, (name, acc) in zip(cols, sorted_accs):
            short = name.replace("🏆 ","")
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='font-size:1.6rem'>{acc:.1f}%</div>
                <div class='metric-label'>{short}</div>
                <div style='margin-top:8px'>{acc_badge(acc)}</div>
            </div>""", unsafe_allow_html=True)

        # ── Cross-validation ──
        st.markdown("<div class='section-title'>📊 5-Fold Cross-Validation</div>", unsafe_allow_html=True)
        cv_df = pd.DataFrame({
            n: {"Mean": f"{s.mean():.2f}%", "Std": f"±{s.std():.2f}%",
                "Min":  f"{s.min():.2f}%",  "Max": f"{s.max():.2f}%"}
            for n, s in cv_scores.items()}).T
        st.dataframe(cv_df, use_container_width=True)

        # ── ROC ──
        st.markdown("<div class='section-title'>📈 ROC Curves</div>", unsafe_allow_html=True)
        trained = {n: r[0] for n, r in results.items()}
        st.pyplot(plot_roc(trained, y_test, X_test)); plt.clf()

        # ── Per-model tabs ──
        st.markdown("<div class='section-title'>🔍 Model Details</div>", unsafe_allow_html=True)
        tabs = st.tabs(list(results.keys()))
        for tab, (name, (model, preds, acc)) in zip(tabs, results.items()):
            with tab:
                c1, c2 = st.columns([1, 1.4])
                with c1:
                    st.pyplot(plot_confusion(y_test, preds, name)); plt.clf()
                with c2:
                    st.dataframe(get_report_df(y_test, preds), use_container_width=True)
                    fig_imp = plot_importance(model, list(X.columns), "Feature Importance")
                    if fig_imp:
                        st.pyplot(fig_imp); plt.clf()

# ─────────────────────────────────────────────
#  PREDICT PAGE
# ─────────────────────────────────────────────
def prediction_page():
    st.markdown("<div class='page-title'>🎯 Risk Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Enter patient metrics to get a personalised diabetes risk score</div>",
                unsafe_allow_html=True)

    if 'trained_models' not in st.session_state:
        st.markdown("""
        <div class='glass-card' style='text-align:center;padding:48px'>
            <div style='font-size:3rem'>🤖</div>
            <div style='font-size:1.2rem;color:#e6edf3;margin:12px 0'>No models trained yet</div>
            <div style='color:#8b949e'>Go to <b style='color:#0ea5e9'>Model Lab</b> and hit Train first.</div>
        </div>""", unsafe_allow_html=True)
        return

    use_fe = st.session_state.get('use_fe', True)

    st.markdown("<div class='section-title'>📝 Patient Metrics</div>", unsafe_allow_html=True)
    with st.form("pred_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose     = st.number_input("Glucose (mg/dL)", 0, 300, 120)
        with c2:
            bp          = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
            skin        = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        with c3:
            insulin     = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
            bmi         = st.number_input("BMI (kg/m²)", 0.0, 70.0, 25.0, step=0.1)
        with c4:
            dpf         = st.number_input("Diabetes Pedigree Fn", 0.0, 3.0, 0.5, step=0.01)
            age         = st.number_input("Age (years)", 0, 120, 30)
        submitted = st.form_submit_button("🔍  Analyse Risk", use_container_width=True)

    if submitted:
        raw_df = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
                              columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                                       'Insulin','BMI','DiabetesPedigreeFunction','Age'])
        if use_fe:
            raw_df['Glucose_BMI']     = raw_df['Glucose'] * raw_df['BMI']
            raw_df['Age_Glucose']     = raw_df['Age']     * raw_df['Glucose']
            raw_df['Insulin_Glucose'] = raw_df['Insulin'] * raw_df['Glucose']
            raw_df['BMI_Age']         = raw_df['BMI']     * raw_df['Age']

        scaled = st.session_state['scaler'].transform(raw_df)

        st.markdown("<div class='section-title'>📊 Model Predictions</div>", unsafe_allow_html=True)
        probs = []
        model_cols = st.columns(len(st.session_state['trained_models']))
        for col, (name, model) in zip(model_cols, st.session_state['trained_models'].items()):
            pred  = model.predict(scaled)[0]
            proba = model.predict_proba(scaled)[0]
            p     = proba[1]
            probs.append(p)
            short = name.replace("🏆 ","")
            color = "#ef4444" if pred == 1 else "#00d4aa"
            label = "DIABETIC" if pred == 1 else "CLEAR"
            col.markdown(f"""
            <div class='metric-card' style='border-color:{color}44'>
                <div style='font-size:.72rem;color:#8b949e;text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:8px'>{short}</div>
                <div style='font-size:1.3rem;font-weight:800;color:{color}'>{label}</div>
                <div style='font-size:.85rem;color:#8b949e;margin-top:4px'>{p*100:.1f}% risk</div>
                <div style='background:#21262d;border-radius:4px;height:5px;margin-top:10px'>
                    <div style='width:{p*100:.1f}%;background:{color};height:5px;border-radius:4px'></div>
                </div>
            </div>""", unsafe_allow_html=True)

        avg = np.mean(probs) * 100
        st.markdown("<div class='section-title'>⚡ Consensus Risk Score</div>", unsafe_allow_html=True)

        if avg < 30:
            lbl, cls, icon, advice = "LOW RISK", "risk-low", "✅", \
                "Your metrics look healthy. Maintain your current lifestyle and schedule regular checkups."
        elif avg < 65:
            lbl, cls, icon, advice = "MODERATE RISK", "risk-medium", "⚠️", \
                "Some markers are elevated. Consider a GP consultation and monitor glucose levels."
        else:
            lbl, cls, icon, advice = "HIGH RISK", "risk-high", "🚨", \
                "Multiple risk factors detected. Please consult a healthcare professional as soon as possible."

        st.markdown(f"""
        <div class='glass-card' style='text-align:center'>
            <div style='font-size:3.5rem;margin-bottom:8px'>{icon}</div>
            <div style='font-size:2.5rem;font-weight:800;margin-bottom:6px'>
                <span class='acc-badge {cls}'>{avg:.1f}%</span></div>
            <div style='font-size:1.2rem;font-weight:700;margin:12px 0'>{lbl}</div>
            <div style='color:#8b949e;max-width:480px;margin:0 auto'>{advice}</div>
        </div>
        <div style='text-align:center;color:#484f58;font-size:.78rem;margin-top:10px'>
            ⚠️ For educational purposes only — not a substitute for medical advice.</div>
        """, unsafe_allow_html=True)

        # Probability bar chart
        st.markdown("<div class='section-title'>📉 Probability by Model</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9,3.5))
        apply_dark(fig)
        names  = [n.replace("🏆 ","") for n in st.session_state['trained_models']]
        colors = ["#ef4444" if p > .5 else ACCENT for p in probs]
        bars   = ax.bar(names, [p*100 for p in probs],
                        color=colors, edgecolor=BORDER, width=0.5)
        ax.axhline(50, color=MUTED, linestyle='--', lw=1, alpha=.6)
        ax.set_ylim(0,105)
        ax.set_ylabel("Diabetic Probability (%)", color=MUTED)
        for bar, val in zip(bars, probs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                    f"{val*100:.1f}%", ha='center', va='bottom',
                    color=TEXT, fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.clf()

# ─────────────────────────────────────────────
#  ROUTING
# ─────────────────────────────────────────────
ensure_dataset()

if   page == "🏠  Home":          home_page()
elif page == "🔬  Data Explorer":  data_exploration_page()
elif page == "🤖  Model Lab":      model_training_page()
elif page == "🎯  Predict":        prediction_page()

st.markdown("""
<div class='footer'>
    DiabetesIQ · Streamlit + scikit-learn + XGBoost ·
    <a href='https://www.kaggle.com/datasets/hossamhassan1/diabetes-classification'
       style='color:#00d4aa;text-decoration:none'>Dataset ↗</a>
</div>""", unsafe_allow_html=True)
