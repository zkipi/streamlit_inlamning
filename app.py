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
# Title & intro
# --------------------------------------------------
st.title("📦 Amazon Sales & Reviews – Data Science Project")
st.markdown("""
Detta projekt analyserar ett Amazon-dataset med fokus på **pris, rabatt och kundbetyg**.
Datasetet innehåller över 1000 olika brands, så visualiseringar är optimerade för att visa
de mest relevanta trenderna utan att bli röriga.
""")

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/amazon_sales_cleaned.csv")

df = load_data()

# Show columns (debug)
st.write("Kolumner i datasetet:")
st.write(df.columns.tolist())

# --------------------------------------------------
# Data cleaning
# --------------------------------------------------
# Rensa priser
for col in ["discount_price", "actual_price"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

# Ta bort rader med noll actual_price
df = df[df["actual_price"] > 0]

# Skapa discount percentage
df["discount_percentage"] = ((df["actual_price"] - df["discount_price"]) / df["actual_price"] * 100)

# Rensa rating
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["no_of_ratings"] = pd.to_numeric(df["no_of_ratings"], errors="coerce")

# Drop rows with missing values
df = df.dropna(subset=["discount_price", "rating"])

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("Filter")

# Prisfilter
min_price, max_price = st.sidebar.slider(
    "Välj prisintervall",
    int(df["discount_price"].min()),
    int(df["discount_price"].max()),
    (int(df["discount_price"].min()), int(df["discount_price"].max()))
)

filtered_df = df[(df["discount_price"] >= min_price) & (df["discount_price"] <= max_price)]

# Top-N brands filter
st.sidebar.markdown("Välj antal topp-brands att visa")
top_n = st.sidebar.slider("Topp N brands", 5, 20, 10)

# Beräkna topp brands baserat på antal produkter
top_brands = (
    filtered_df["brand"]
    .value_counts()
    .nlargest(top_n)
    .index.tolist()
)

filtered_top_df = filtered_df[filtered_df["brand"].isin(top_brands)]

# --------------------------------------------------
# Data overview
# --------------------------------------------------
st.subheader("Översikt av datan")
col1, col2 = st.columns(2)
with col1:
    st.write("Antal produkter:", filtered_df.shape[0])
with col2:
    st.write("Antal brands totalt:", df["brand"].nunique())
    st.write(f"Topp {top_n} brands visas i visualiseringarna.")

# --------------------------------------------------
# Visualization 1: Prisfördelning
# --------------------------------------------------
st.subheader("Prisfördelning")

fig, ax = plt.subplots()
sns.histplot(filtered_df["discount_price"], bins=40, kde=False, ax=ax)
ax.set_xlabel("Rabatterat pris")
ax.set_ylabel("Antal produkter")
st.pyplot(fig)

# --------------------------------------------------
# Visualization 2: Top brands med högsta genomsnittlig rabatt
# --------------------------------------------------
st.subheader("Top brands med högst genomsnittlig rabatt")

discount_by_brand = (
    filtered_top_df.groupby("brand")["discount_percentage"]
    .mean()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 5))
discount_by_brand.plot(kind="bar", ax=ax)
ax.set_ylabel("Genomsnittlig rabatt (%)")
ax.set_xlabel("Brand")
plt.xticks(rotation=45)
st.pyplot(fig)

# --------------------------------------------------
# Visualization 3: Pris vs rating
# --------------------------------------------------
st.subheader("Pris vs kundbetyg")

fig, ax = plt.subplots()
ax.scatter(filtered_df["discount_price"], filtered_df["rating"], alpha=0.4)
ax.set_xlabel("Rabatterat pris")
ax.set_ylabel("Rating")
st.pyplot(fig)

st.markdown("""
Det finns inget starkt samband mellan pris och betyg, vilket tyder på att
pris inte är den viktigaste faktorn för kundnöjdhet.
""")

# --------------------------------------------------
# Visualization 4: Antal produkter per topp-brand
# --------------------------------------------------
st.subheader("Antal produkter per topp-brand")

count_by_brand = filtered_top_df["brand"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
count_by_brand.plot(kind="bar", ax=ax)
ax.set_ylabel("Antal produkter")
ax.set_xlabel("Brand")
plt.xticks(rotation=45)
st.pyplot(fig)

# --------------------------------------------------
# Conclusion
# --------------------------------------------------
st.subheader("Slutsats")

st.markdown(f"""
- De flesta produkter ligger i ett lågt till medelhögt prisintervall.
- Rabattnivåer varierar mellan topp {top_n} brands.
- Sambandet mellan pris och kundbetyg är svagt.
- Vissa brands dominerar antalet produkter, men visualiseringarna fokuserar på topp {top_n} brands
  för att göra trenderna tydliga.
""")
