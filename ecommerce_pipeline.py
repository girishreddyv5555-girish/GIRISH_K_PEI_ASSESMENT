# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # E-commerce Sales Data Processing
# MAGIC
# MAGIC | Layer | Tables |
# MAGIC |---|---|
# MAGIC | Raw (bronze) | `raw_products`, `raw_orders`, `raw_customers` |
# MAGIC | Enriched (silver) | `enriched_customers`, `enriched_products`, `enriched_orders` |
# MAGIC | Aggregate (gold) | `agg_profit` (Year x Category x Sub-Category x Customer) |
# MAGIC
# MAGIC Change `source_dir` to match the volume where your raw files are stored. In my case, the raw files are under `/Volumes/workspace/ecommerce/landing`.
# MAGIC
# MAGIC **Unit Test Cases File : test_ecommerce_transform.py** 

# COMMAND ----------

# MAGIC %md ##Configuration

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
import pandas as pd
from pyspark.sql import functions as F

sys.path.append(os.getcwd())
import ecommerce_transforms as T

dbutils.widgets.text("source_dir", "/Volumes/workspace/ecommerce/landing", "Folder with the 3 source files")
dbutils.widgets.text("target_schema", "ecommerce", "Target schema (database)")

SOURCE_DIR = dbutils.widgets.get("source_dir")
SCHEMA = dbutils.widgets.get("target_schema")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")


def save_table(df, name):
    """Write a Delta table (idempotent overwrite) and return the fully-qualified name."""
    fq = f"{SCHEMA}.{name}"
    (df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq))
    print(f"✔ {fq:<35} {spark.table(fq).count():>7,} rows")
    return fq

# COMMAND ----------

# MAGIC %md ## Raw tables

# COMMAND ----------

# MAGIC %pip install openpyxl -q

# COMMAND ----------

#Products.csv
products_src = (spark.read
                .option("header", True)
                .option("inferSchema", True)
                .option("quote", '"')
                .option("escape", '"')       
                .option("multiLine", True)   
                .csv(f"{SOURCE_DIR}/Products.csv"))

#Orders.json
orders_src = spark.read.option("multiLine", True).json(f"{SOURCE_DIR}/Orders.json")

#Customer.xlsx
pdf = pd.read_excel(f"{SOURCE_DIR}/Customer.xlsx", dtype=str)
pdf = pdf.astype(object).where(pdf.notna(), None)
customers_src = spark.createDataFrame(pdf)

raw_products_fq = save_table(T.add_ingest_metadata(T.standardise_columns(products_src), "Products.csv"), "raw_products")
raw_orders_fq = save_table(T.add_ingest_metadata(T.standardise_columns(orders_src), "Orders.json"), "raw_orders")
raw_customers_fq = save_table(T.add_ingest_metadata(T.standardise_columns(customers_src), "Customer.xlsx"), "raw_customers")

# COMMAND ----------

# MAGIC %md ##Enriched customers,products and orders

# COMMAND ----------

enriched_customers_fq = save_table(T.enrich_customers(spark.table(raw_customers_fq)), "enriched_customers")
enriched_products_fq = save_table(T.enrich_products(spark.table(raw_products_fq)), "enriched_products")
enriched_orders_df = T.build_enriched_orders(
    spark.table(raw_orders_fq),
    spark.table(enriched_customers_fq),
    spark.table(enriched_products_fq),
)
enriched_orders_fq = save_table(enriched_orders_df, "enriched_orders")

# COMMAND ----------

# MAGIC %md ##Aggregate table: profit by Year, Category, Sub-Category, Customer

# COMMAND ----------

agg_fq = save_table(T.build_profit_aggregate(spark.table(enriched_orders_fq)), "agg_profit")

# COMMAND ----------

# MAGIC %md ## Data-quality 

# COMMAND ----------

raw_n = spark.table(raw_orders_fq).count()
raw_dedup_n = T.clean_orders(spark.table(raw_orders_fq)).count()
enr = spark.table(enriched_orders_fq)
enr_n = enr.count()

checks = {
    "no order lines lost or fanned-out by joins": enr_n == raw_dedup_n,
    "no NULL order dates / years": enr.filter("order_date IS NULL OR order_year IS NULL").count() == 0,
    "every customer matched": enr.filter("customer_id IS NOT NULL AND customer_country IS NULL").count() == 0,
    "profit has at most 2 decimals": enr.filter(F.col("profit") != F.round("profit", 2)).count() == 0,
    "aggregate profit == enriched profit": (
        spark.table(agg_fq).agg(F.round(F.sum("total_profit"), 2)).first()[0]
        == enr.agg(F.round(F.sum("profit"), 2)).first()[0]),
    "aggregate grain is unique": (
        spark.table(agg_fq).count() == spark.table(agg_fq).dropDuplicates(T.AGG_KEYS).count()),
}
for name, ok in checks.items():
    print(("✔ " if ok else "✘ ") + name)
assert all(checks.values()), "Data-quality gate failed - see above"

print(f"\nraw order lines: {raw_n:,} | duplicates removed: {raw_n - raw_dedup_n} | enriched: {enr_n:,}")
print(f"order lines with product not in product master: {enr.filter('NOT product_found').count()}")
print(f"customers with NULL name: {spark.table(enriched_customers_fq).filter('customer_name IS NULL').count()}")

# COMMAND ----------

# MAGIC %md ##SQL aggregates

# COMMAND ----------

def run_sql(name):
    return spark.sql(T.SQL_QUERIES[name].format(t=agg_fq))

#Profit by Year
display(run_sql("profit_by_year"))

# COMMAND ----------

#Profit by Year + Product Category
display(run_sql("profit_by_year_category"))

# COMMAND ----------

#Profit by Customer
display(run_sql("profit_by_customer"))

# COMMAND ----------

#Profit by Customer + Year
display(run_sql("profit_by_customer_year"))

# COMMAND ----------

# MAGIC %md ## Unit tests
# MAGIC Runs `test_ecommerce_transforms.py`

# COMMAND ----------

# MAGIC %pip install pytest -q

# COMMAND ----------

import os, sys, pytest
sys.dont_write_bytecode = True
exit_code = pytest.main(["-v", "-p", "no:cacheprovider", os.path.join(os.getcwd(), "test_ecommerce_transform.py")])
assert exit_code == 0, "Unit tests failed"