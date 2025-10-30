# streamlitVID.py — Year (global), Category per Tab, Product bar chart if >1 category
# run: streamlit run streamlitVID.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

# ---------- Helpers ----------
def load_data_any(base_dir: Path) -> pd.DataFrame:
    csv_path  = base_dir / "Copy of finalProj_df - df.csv"
    xlsx_path = base_dir / "Copy of finalProj_df.xlsx"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif xlsx_path.exists():
        df = pd.read_excel(xlsx_path)
    else:
        st.error(f"File tidak ditemukan: {csv_path.name} / {xlsx_path.name}")
        st.stop()

    df["order_date"] = pd.to_datetime(df.get("order_date"), errors="coerce", infer_datetime_format=True)
    df["revenue"] = df.get("after_discount", df.get("revenue"))
    df["cost"]    = df.get("cogs", 0)
    df["profit"]  = df["revenue"] - df["cost"]

    if "category" not in df.columns: df["category"] = "Unknown"
    if "payment_method" not in df.columns: df["payment_method"] = "Unknown"
    if "id" not in df.columns: df["id"] = np.arange(len(df))
    if "qty_ordered" not in df.columns: df["qty_ordered"] = 1
    if "customer_id" not in df.columns:
        for c in ["user_id","buyer_id","email","account_id"]:
            if c in df.columns:
                df["customer_id"] = df[c]; break
        if "customer_id" not in df.columns:
            df["customer_id"] = df["id"]
    return df

def build_product_label(df: pd.DataFrame) -> pd.Series | None:
    for c in ["product_name","item_name","product","sku","product_id"]:
        if c in df.columns:
            return df[c].astype(str).where(df[c].notna(),"Unknown")
    for a,b in [("brand","model"),("category","sub_category"),("sku","variant")]:
        if a in df.columns and b in df.columns:
            return (df[a].astype(str).fillna("Unknown")+" "+df[b].astype(str).fillna("")).str.strip()
    if "category" in df.columns: return df["category"].astype(str)
    return None

def monthly_agg(dfx: pd.DataFrame, selected_year: int|None) -> pd.DataFrame:
    m = (dfx.dropna(subset=["order_date"])
           .assign(month=lambda x: x["order_date"].dt.to_period("M").dt.to_timestamp())
           .groupby("month", as_index=False)
           .agg(revenue=("revenue","sum"),
                profit =("profit","sum"),
                orders =("id","nunique")))
    if selected_year is not None:
        months = pd.date_range(f"{selected_year}-01-01", f"{selected_year}-12-01", freq="MS")
        m = pd.DataFrame({"month":months}).merge(m, on="month", how="left").fillna(0)
    m["aov"] = np.where(m["orders"]>0, m["revenue"]/m["orders"], 0.0)
    return m.sort_values("month")

def category_top_table(dfx: pd.DataFrame, topn=8) -> pd.DataFrame:
    t = (dfx.groupby("category", dropna=False)
           .agg(revenue=("revenue","sum"),
                profit =("profit","sum"),
                orders =("id","nunique"))
           .reset_index()
           .sort_values("revenue", ascending=False)
           .head(topn))
    t["aov"] = np.where(t["orders"]>0, t["revenue"]/t["orders"], np.nan)
    return t

# ---------- App ----------
def main():
    st.set_page_config(page_title="Sales & Profit Dashboard", page_icon="📊", layout="wide")
    st.title("📊 Sales & Profit Dashboard")

    BASE_DIR = Path(__file__).resolve().parent
    df = load_data_any(BASE_DIR)

    # ======= TOP FILTER BAR (Year only) =======
    fb1, fb_spacer = st.columns([1, 3], gap="large")
    with fb1:
        years = sorted([int(y) for y in df["order_date"].dt.year.dropna().unique()])
        year = st.selectbox("Year", options=["All"]+years,
                            index=(years.index(2022)+1 if 2022 in years else 0))
    # Apply year filter (global)
    dff_year = df if year == "All" else df[df["order_date"].dt.year == int(year)]
    selected_year = None if year=="All" else int(year)

    st.markdown("---")

    # Tabs
    tab_overview, tab_product = st.tabs(["🧭 Overview", "🛒 Product"])

    # ============ OVERVIEW ============
    with tab_overview:
        # Page-level Category (single dropdown)
        cat_opts = ["All"] + sorted(dff_year["category"].astype(str).dropna().unique().tolist())
        sel_cat_over = st.selectbox("Category (Overview)", options=cat_opts, index=0)
        dff_over = dff_year if sel_cat_over=="All" else dff_year[dff_year["category"].astype(str)==sel_cat_over]

        # KPI row
        total_revenue = float(dff_over["revenue"].sum()) if not dff_over.empty else 0.0
        total_profit  = float(dff_over["profit"].sum())  if not dff_over.empty else 0.0
        orders        = int(dff_over["id"].nunique())    if not dff_over.empty else 0
        aov_val       = (total_revenue/orders) if orders else 0
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Values Sales", f"{total_revenue:,.0f}")
        c2.metric("Net Profit", f"{total_profit:,.0f}")
        c3.metric("AOV", f"{aov_val:,.0f}")
        c4.metric("Orders", f"{orders:,}")

        # Category Summary (Top 8)
        st.subheader("Category Summary (Top 8)")
        cat_tbl = category_top_table(dff_over)
        if cat_tbl.empty:
            st.info("Tidak ada data kategori untuk filter saat ini.")
        else:
            show = cat_tbl.rename(columns={
                "category":"Category","revenue":"Values Sales",
                "profit":"Net Profit","orders":"Orders","aov":"AOV"
            }).copy()
            for col in ["Values Sales","Net Profit","AOV"]:
                show[col] = show[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
            st.dataframe(show, use_container_width=True)

        st.markdown("---")

        # Combo chart: Values Sales vs Net Profit + AOV
        st.subheader("Values Sales vs Net Profit and AOV (Monthly)")
        m = monthly_agg(dff_over, selected_year)
        if m.empty:
            st.info("Tidak ada data bulanan.")
        else:
            fig = go.Figure()
            fig.add_bar(x=m["month"], y=m["revenue"], name="Values Sales")
            fig.add_bar(x=m["month"], y=m["profit"],  name="Net Profit")
            fig.add_scatter(x=m["month"], y=m["aov"], name="AOV",
                            mode="lines+markers", yaxis="y2", line=dict(color="red"))
            fig.update_layout(
                barmode="group", height=460, margin=dict(l=10,r=10,t=30,b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(title="Sales / Profit"),
                yaxis2=dict(title="AOV", overlaying="y", side="right"),
                xaxis=dict(title="Month")
            )
            st.plotly_chart(fig, use_container_width=True)

    # ============ PRODUCT ============
    with tab_product:
        # Page-level Category (multiselect)
        cat_list = sorted(dff_year["category"].astype(str).dropna().unique().tolist())
        sel_cat_prod = st.multiselect("Category (Product)", options=cat_list, default=cat_list)

        dff_prod_base = dff_year[dff_year["category"].astype(str).isin(sel_cat_prod)] if sel_cat_prod else dff_year

        # Product filter (multiselect; empty => all)
        prod_label = build_product_label(dff_prod_base)
        if prod_label is None:
            st.info("Kolom produk tidak ditemukan (product_name/sku/product_id/brand+model, dll).")
            return
        dff_prod_base = dff_prod_base.assign(_product_label=prod_label)

        ranked = (dff_prod_base.groupby("_product_label", dropna=False)
                    .agg(revenue=("revenue","sum"))
                    .reset_index()
                    .sort_values("revenue", ascending=False))
        prod_options = ranked["_product_label"].astype(str).tolist()
        sel_products = st.multiselect("Pilih Produk (kosongkan untuk semua)", options=prod_options, default=[])

        view = dff_prod_base if len(sel_products)==0 else dff_prod_base[dff_prod_base["_product_label"].isin(sel_products)]

        # KPIs
        rev = float(view["revenue"].sum()) if not view.empty else 0.0
        ords= int(view["id"].nunique()) if not view.empty else 0
        qty = int(view["qty_ordered"].sum()) if not view.empty else 0
        cus = int(view["customer_id"].nunique()) if not view.empty else 0
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Revenue", f"{rev:,.0f}")
        k2.metric("Orders", f"{ords:,}")
        k3.metric("Quantity", f"{qty:,}")
        k4.metric("Customers", f"{cus:,}")

        # Payment pie + table
        if view.empty:
            st.info("Tidak ada data untuk kombinasi filter ini.")
        else:
            pay = (view.groupby("payment_method", dropna=False)
                     .agg(revenue=("revenue","sum"), orders=("id","nunique"))
                     .reset_index()
                     .sort_values("revenue", ascending=False))
            colL, colR = st.columns([1.2,1.8], gap="large")
            with colL:
                st.markdown("**Payment Methods — Pie (by Revenue)**")
                fig_pie = go.Figure(data=[go.Pie(labels=pay["payment_method"],
                                                 values=pay["revenue"], hole=.45)])
                fig_pie.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig_pie, use_container_width=True)
            with colR:
                tbl = pay.rename(columns={"payment_method":"Payment Method"})
                tbl["revenue"] = tbl["revenue"].apply(lambda x: f"{x:,.0f}")
                st.dataframe(tbl, use_container_width=True, height=420)

            st.markdown("---")

            # ===== Horizontal bar jika kategori > 1 =====
            # User pilih metrik yang mau dilihat
            metric = st.radio("Metric untuk chart kategori", ["Revenue","Orders","Quantity"],
                              horizontal=True, index=0, key="cat_metric")
            agg_map = {
                "Revenue": ("revenue","sum"),
                "Orders" : ("id","nunique"),
                "Quantity": ("qty_ordered","sum"),
            }
            col_name, how = agg_map[metric]

            # Ambil subset kategori yang aktif (dari sel_cat_prod)
            active_cats = sel_cat_prod if sel_cat_prod else cat_list
            cat_subset = view[view["category"].astype(str).isin(active_cats)]

            # Jika lebih dari satu kategori -> tampilkan bar chart horizontal
            if len(set(cat_subset["category"].astype(str))) > 1:
                cat_agg = (cat_subset.groupby("category", dropna=False)
                                      .agg(val=(col_name, how))
                                      .reset_index()
                                      .sort_values("val", ascending=True))
                fig_bar = go.Figure()
                fig_bar.add_bar(y=cat_agg["category"], x=cat_agg["val"], orientation="h",
                                name=metric)
                fig_bar.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10),
                                      xaxis_title=metric, yaxis_title="Category")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.caption("Tip: pilih lebih dari satu kategori untuk melihat perbandingan horizontal bar.")

if __name__ == "__main__":
    main()
