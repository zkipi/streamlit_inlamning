import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Amazon Sales & Reviews Analysis",
    layout="wide"
)

# --------------------------------------------------
# Title & introduction
# --------------------------------------------------
st.title("📦 Amazon Sales & Reviews – Data Science Project")

st.markdown("""
Detta projekt analyserar ett Amazon-dataset med fokus på **pris, rabatter och kundbetyg**.
Syftet är att undersöka om det finns samband mellan prisnivåer, rabatter och hur kunder
betygsätter produkter i olika kategorier.
""")

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/amazon_sales_cleaned.csv")

df = load_data()

# --------------------------------------------------
# Data cleaning
# --------------------------------------------------
price_cols = ["discount_price", "actual_price"]

for col in price_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

df["discount_percentage"] = (
    (df["actual_price"] - df["discount_price"]) / df["actual_price"] * 100
)

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["no_of_ratings"] = pd.to_numeric(df["no_of_ratings"], errors="coerce")

df = df.dropna(subset=["discount_price", "rating"])

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("Filter")

categories = st.sidebar.multiselect(
    "Välj produktkategorier",
    options=sorted(df["brand"].unique()),
    default=sorted(df["brand"].unique())
)

min_price, max_price = st.sidebar.slider(
    "Välj prisintervall",
    int(df["discount_price"].min()),
    int(df["discount_price"].max()),
    (
        int(df["discount_price"].min()),
        int(df["discount_price"].max())
    )
)

filtered_df = df[
    (df["brand"].isin(categories)) &
    (df["discount_price"] >= min_price) &
    (df["discount_price"] <= max_price)
]

# --------------------------------------------------
# Data overview
# --------------------------------------------------
st.subheader("Översikt av datan")

col1, col2 = st.columns(2)
with col1:
    st.write("Antal produkter:", filtered_df.shape[0])
with col2:
    st.write("Antal kategorier:", filtered_df["brand"].nunique())

# --------------------------------------------------
# Visualization 1: Price distribution
# --------------------------------------------------
st.subheader("Prisfördelning")

fig, ax = plt.subplots()
sns.histplot(filtered_df["discount_price"], bins=40, ax=ax)
ax.set_xlabel("Rabatterat pris")
ax.set_ylabel("Antal produkter")

st.pyplot(fig)

st.markdown("""
Diagrammet visar att de flesta produkter säljs inom ett relativt lågt till
medelhögt prisspann, med ett mindre antal dyrare produkter.
""")

# --------------------------------------------------
# Visualization 2: Average discount by category
# --------------------------------------------------
st.subheader("Genomsnittlig rabatt per kategori")

discount_by_category = (
    filtered_df
    .groupby("brand")["discount_percentage"]
    .mean()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(8, 5))
discount_by_category.plot(kind="bar", ax=ax)
ax.set_ylabel("Genomsnittlig rabatt (%)")
ax.set_xlabel("Kategori")

st.pyplot(fig)

st.markdown("""
Vissa kategorier erbjuder betydligt högre rabatter än andra, vilket kan
indikera hårdare konkurrens eller kampanjstrategier.
""")

# --------------------------------------------------
# Visualization 3: Price vs rating
# --------------------------------------------------
st.subheader("Samband mellan pris och betyg")

fig, ax = plt.subplots()
ax.scatter(
    filtered_df["discount_price"],
    filtered_df["rating"],
    alpha=0.4
)

ax.set_xlabel("Rabatterat pris")
ax.set_ylabel("Betyg")

st.pyplot(fig)

st.markdown("""
Det finns inget tydligt starkt samband mellan pris och betyg. Dyrare produkter
har inte nödvändigtvis högre betyg än billigare produkter.
""")

# --------------------------------------------------
# Visualization 4: Review count vs rating
# --------------------------------------------------
st.subheader("Antal recensioner och betyg")

fig, ax = plt.subplots()
ax.scatter(
    filtered_df["no_of_ratings"],
    filtered_df["rating"],
    alpha=0.3
)

ax.set_xlabel("Antal recensioner")
ax.set_ylabel("Betyg")

st.pyplot(fig)

st.markdown("""
Produkter med fler recensioner tenderar att ha mer stabila betyg, medan produkter
med få recensioner uppvisar större variation.
""")

# --------------------------------------------------
# Conclusion
# --------------------------------------------------
st.subheader("Sammanfattning och slutsatser")

st.markdown("""
- De flesta produkter på Amazon säljs till relativt låga priser.
- Rabattnivåer varierar kraftigt mellan produktkategorier.
- Priset verkar inte vara den avgörande faktorn för kundbetyg.
- Antalet recensioner påverkar hur stabilt ett betyg är.

Denna analys visar att kundnöjdhet är ett komplext samspel mellan flera faktorer,
inte enbart pris eller rabatt.
""")
