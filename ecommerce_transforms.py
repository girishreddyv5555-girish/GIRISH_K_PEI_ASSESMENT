"""
ecommerce_transforms.py
=======================
Pure PySpark transformation functions for the e-commerce sales pipeline.

Design rules
------------
* Every function takes DataFrame(s) and returns a DataFrame - no reads, no
  writes, no globals. That makes each step unit-testable with tiny in-memory
  DataFrames (see test_ecommerce_transforms.py).
* I/O (CSV / JSON / XLSX ingestion, Delta writes) lives in the notebook.

Data-quality findings that drive the logic (profiled on the supplied files)
--------------------------------------------------------------------------
1. Customer.xlsx  - 91 names carry junk (digits, punctuation, stray spaces,
   e.g. 'Tam&^*ara Willing___)ham'); 8 names are NULL; 'phone' has mixed
   int/str types. Names containing legitimate accents (Bühler) must survive.
2. Products.csv   - `Product ID` is NOT unique (1,850 rows / 1,818 ids): the
   file is one row per (product, state) and 32 ids carry different product
   names in different states. Category / Sub-Category are consistent per id,
   so a (product_id, category, sub_category) dimension is 1 row per id.
   The `State` column cannot be used to join to orders (only ~15% match).
3. Orders.json    - dates are 'd/M/yyyy' strings; 204 order lines (44 product
   ids) have no matching product; 1 exact duplicate line (same values,
   different Row ID); every customer id matches Customer.xlsx.
   `Price` is the line total (Quantity x unit price), not a unit price.
"""
import re
from typing import List

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

UNKNOWN = "Unknown"


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
def to_snake_case(name: str) -> str:
    """'Sub-Category' -> 'sub_category', 'Price per product' -> 'price_per_product'."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()


def standardise_columns(df: DataFrame) -> DataFrame:
    """Rename all columns to snake_case (Delta-friendly; no spaces/hyphens)."""
    return df.toDF(*[to_snake_case(c) for c in df.columns])


def add_ingest_metadata(df: DataFrame, source_name: str) -> DataFrame:
    """Audit columns for raw/bronze tables."""
    return (df.withColumn("_source_file", F.lit(source_name))
              .withColumn("_ingested_at", F.current_timestamp()))


def clean_person_name(col: F.Column) -> F.Column:
    """
    Remove noise from a person name while keeping real letters (incl. accents),
    apostrophes (O'Rourke) and hyphens (Bell-Smith).

    Rules, applied in order:
      1. hyphen/apostrophe not sitting between two letters is junk -> removed
      2. a run of non-letters that contains junk and is followed by a
         lower-case letter is *inside a word*  -> removed ('Tam&^*ara' -> 'Tamara')
      3. 2+ spaces inside a word (followed by lower-case) -> removed ('B    ecky')
      4. any remaining junk run (word boundary) -> single space
      5. collapse whitespace and trim; empty -> NULL

    Known limitation: digits that *replaced* a letter ('Bi1l', 'N0ra') cannot be
    recovered reliably, so those digits are dropped (affects ~7 names).
    """
    c = F.regexp_replace(col, r"(?<!\p{L})['\-]|['\-](?!\p{L})", "")
    c = F.regexp_replace(c, r"[^\p{L}]*[^\p{L} ][^\p{L}]*(?=\p{Ll})", "")
    c = F.regexp_replace(c, r"(?<=\p{L}) {2,}(?=\p{Ll})", "")
    c = F.regexp_replace(c, r"[^\p{L}'\-]*[^\p{L}'\- ][^\p{L}'\-]*", " ")
    c = F.trim(F.regexp_replace(c, r"\s+", " "))
    return F.when(col.isNull() | (F.length(c) == 0), F.lit(None)).otherwise(c)


# --------------------------------------------------------------------------
# Task 2: enriched customers / products
# --------------------------------------------------------------------------
def enrich_customers(raw_customers: DataFrame) -> DataFrame:
    """
    Clean customer master data.
      * snake_case already applied at raw layer
      * customer_name cleaned (junk removed, NULL preserved)
      * text columns trimmed; postal_code cast to string (keeps leading zeros)
      * phone cast to string (source mixes int and str)
      * de-duplicated on customer_id (defensive - source has none today)
    """
    df = (raw_customers
          .withColumn("customer_name", clean_person_name(F.col("customer_name")))
          .withColumn("customer_id", F.trim("customer_id"))
          .withColumn("postal_code", F.col("postal_code").cast("string"))
          .withColumn("phone", F.col("phone").cast("string")))
    for c in ("segment", "country", "city", "state", "region"):
        df = df.withColumn(c, F.trim(F.col(c)))
    return df.dropDuplicates(["customer_id"])


def enrich_products(raw_products: DataFrame) -> DataFrame:
    """
    Clean product master data. Grain is kept as the source grain
    (product_id, state) because name and price legitimately vary by state.
      * all text columns trimmed (3 product names have stray whitespace)
      * price_per_product cast to double and rounded to 2 dp
      * exact duplicate rows removed
    """
    df = raw_products
    for c in ("product_id", "category", "sub_category", "product_name", "state"):
        df = df.withColumn(c, F.trim(F.col(c)))
    df = df.withColumn("price_per_product",
                       F.round(F.col("price_per_product").cast("double"), 2))
    return df.dropDuplicates(["product_id", "state", "product_name", "price_per_product"])


def product_dimension(enriched_products: DataFrame) -> DataFrame:
    """
    One row per product_id carrying category and sub_category.

    Joining orders to the raw product table would multiply rows (a product id
    appears once per state), inflating profit. Category/Sub-Category are
    consistent per id, so distinct() collapses to exactly one row per id.
    If they ever conflict, the alphabetically-first pair wins deterministically.
    """
    w = Window.partitionBy("product_id").orderBy("category", "sub_category")
    return (enriched_products
            .select("product_id", "category", "sub_category")
            .distinct()
            .withColumn("_rn", F.row_number().over(w))
            .filter(F.col("_rn") == 1)
            .drop("_rn"))


# --------------------------------------------------------------------------
# Task 3: enriched orders
# --------------------------------------------------------------------------
def _parse_date(col_name: str) -> F.Column:
    """Parse 'd/M/yyyy'. Invalid values become NULL (never raise, even with ANSI mode on)."""
    return F.to_date(F.try_to_timestamp(F.col(col_name), F.lit("d/M/yyyy")))


def clean_orders(raw_orders: DataFrame, dedupe: bool = True) -> DataFrame:
    """
    Type-cast and clean order lines.
      * order_date / ship_date parsed from 'd/M/yyyy' to DateType (invalid -> NULL)
      * numeric columns cast; profit rounded to 2 dp  (requirement 3.1.1)
      * order_year derived from order_date
      * IDs trimmed
      * optional removal of exact duplicate lines (same values, different row_id)
    """
    df = (raw_orders
          .withColumn("order_id", F.trim("order_id"))
          .withColumn("customer_id", F.trim("customer_id"))
          .withColumn("product_id", F.trim("product_id"))
          .withColumn("order_date", _parse_date("order_date"))
          .withColumn("ship_date", _parse_date("ship_date"))
          .withColumn("quantity", F.col("quantity").cast("int"))
          .withColumn("price", F.col("price").cast("double"))
          .withColumn("discount", F.col("discount").cast("double"))
          .withColumn("profit", F.round(F.col("profit").cast("double"), 2))
          .withColumn("order_year", F.year("order_date")))
    if dedupe:
        business_cols = [c for c in df.columns
                         if c not in ("row_id", "_source_file", "_ingested_at")]
        df = df.dropDuplicates(business_cols)
    return df


def build_enriched_orders(raw_orders: DataFrame,
                          enriched_customers: DataFrame,
                          enriched_products: DataFrame,
                          dedupe: bool = True) -> DataFrame:
    """
    Order information + customer name/country + product category/sub-category.

    LEFT joins are used so no order (and no profit) is ever lost. Orders whose
    product id is missing from the product master get category / sub_category
    = 'Unknown' (flagged by `product_found = false`).
    """
    orders = clean_orders(raw_orders, dedupe=dedupe)
    cust = enriched_customers.select("customer_id", "customer_name", F.col("country").alias("customer_country"))
    prod = product_dimension(enriched_products).withColumn("product_found", F.lit(True))

    return (orders
            .join(cust, "customer_id", "left")
            .join(prod, "product_id", "left")
            .withColumn("product_found", F.coalesce("product_found", F.lit(False)))
            .withColumn("category", F.coalesce("category", F.lit(UNKNOWN)))
            .withColumn("sub_category", F.coalesce("sub_category", F.lit(UNKNOWN)))
            .select("row_id", "order_id", "order_date", "order_year", "ship_date", "ship_mode",
                    "customer_id", "customer_name", "customer_country",
                    "product_id", "category", "sub_category", "product_found",
                    "quantity", "price", "discount", "profit"))


# --------------------------------------------------------------------------
# Task 4: aggregate table
# --------------------------------------------------------------------------
AGG_KEYS: List[str] = ["order_year", "category", "sub_category", "customer_id", "customer_name"]


def build_profit_aggregate(enriched_orders: DataFrame) -> DataFrame:
    """Profit by Year x Category x Sub-Category x Customer (rounded to 2 dp)."""
    return (enriched_orders
            .groupBy(*AGG_KEYS)
            .agg(F.round(F.sum("profit"), 2).alias("total_profit"),
                 F.count("*").alias("order_lines")))


# --------------------------------------------------------------------------
# Task 5: SQL aggregates (run with spark.sql on the aggregate table)
# `{t}` is replaced by the fully-qualified aggregate table / temp view name.
# --------------------------------------------------------------------------
SQL_QUERIES = {
    "profit_by_year": """
        SELECT order_year, ROUND(SUM(total_profit), 2) AS profit
        FROM {t}
        GROUP BY order_year
        ORDER BY order_year""",
    "profit_by_year_category": """
        SELECT order_year, category, ROUND(SUM(total_profit), 2) AS profit
        FROM {t}
        GROUP BY order_year, category
        ORDER BY order_year, category""",
    "profit_by_customer": """
        SELECT customer_id, customer_name, ROUND(SUM(total_profit), 2) AS profit
        FROM {t}
        GROUP BY customer_id, customer_name
        ORDER BY profit DESC, customer_id""",
    "profit_by_customer_year": """
        SELECT customer_id, customer_name, order_year, ROUND(SUM(total_profit), 2) AS profit
        FROM {t}
        GROUP BY customer_id, customer_name, order_year
        ORDER BY customer_id, order_year""",
}
