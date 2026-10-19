# E-commerce Sales Data Processing with Databricks

## Overview

This project builds a small data pipeline in Databricks using PySpark and Delta tables. It takes three raw files (products, orders and customers), cleans them, joins them together, and produces profit reports for the business.

The pipeline follows a simple three-layer structure:

| Layer | Tables | What happens here |
|---|---|---|
| Raw | `raw_products`, `raw_orders`, `raw_customers` | Source data loaded as-is, with tidy column names |
| Enriched | `enriched_customers`, `enriched_products`, `enriched_orders` | Cleaned data, joined and ready for analysis |
| Aggregate | `agg_profit` | Profit by Year, Category, Sub-Category and Customer |

## Files

| File | Purpose |
|---|---|
| `ecommerce_pipeline.py` | The Databricks notebook. Runs all five tasks, the data-quality checks and the unit tests. |
| `ecommerce_transforms.py` | All the transformation logic, written as small functions that take a DataFrame and return a DataFrame. |
| `test_ecommerce_transforms.py` | Unit tests (pytest) for the transformation functions. |

I kept the logic in a separate module on purpose. Functions with no file reads or writes inside them are easy to test with tiny in-memory DataFrames, and the notebook stays short and readable.

## How to run

1. Upload `Products.csv`, `Orders.json` and `Customer.xlsx` to a Unity Catalog volume. In my case this is `/Volumes/workspace/ecommerce/landing`.
2. Put the three `.py` files together in one workspace folder.
3. Open `ecommerce_pipeline.py` and attach a cluster.
4. Set the two widgets at the top:
   - `source_dir`: the volume path from step 1.
   - `target_schema`: for example `workspace.ecommerce`.
5. Click **Run all**. The unit tests run in the last cell.

To run the tests on your own machine instead, use `pip install pyspark pytest` and then `pytest -v`.

## What the notebook does, task by task

1. **Raw tables.** Each source file is read and saved as its own Delta table. Column names are converted to snake_case, and two audit columns (source file and load time) are added. Nothing is cleaned at this stage.
2. **Enriched customers and products.** Customer names are cleaned, text is trimmed, and types are fixed.
3. **Enriched orders.** Order details are joined with customer name and country, and with product category and sub-category. Profit is rounded to 2 decimal places.
4. **Aggregate table.** Total profit by year, category, sub-category and customer.
5. **SQL aggregates.** Four `spark.sql` queries on the aggregate table: profit by year, by year and category, by customer, and by customer and year.

## Data quality issues I found, and what I did about them

I profiled the files before writing any code. These are the problems that affected the design.

**1. Product IDs are not unique in `Products.csv`.**
The file has 1,850 rows but only 1,818 distinct product IDs, because it holds one row per product and state. Joining orders straight onto it would duplicate order lines and inflate profit. Category and sub-category are consistent for each product ID, so I join to a one-row-per-product lookup instead. I also tried joining on state, but only about 15% of orders matched, so that was not an option.

**2. Some orders refer to products that are not in the product file.**
204 order lines (44 product IDs) have no match. I kept them and labelled their category and sub-category as `Unknown`, with a `product_found = false` flag. Dropping them would have lost real profit.

**3. One order line is an exact duplicate.**
It is identical to another line in every column except `Row ID`, so I removed it. This is controlled by a `dedupe` parameter if you would rather keep it.

**4. Customer names contain junk.**
91 names have stray digits, symbols or spaces (for example `Tam&^*ara Willing___)ham`). I clean these with regular expressions that keep genuine accents, apostrophes and hyphens. I checked the result against the initials in each customer ID, and it matched for nearly every name. 8 names are empty in the source and stay empty.

A small number of names (around 7) have a digit in place of a letter, such as `Bi1l`. These cannot be fixed reliably, so the digit is just dropped.

**5. Smaller fixes.**
- Dates arrive as text in `d/M/yyyy` format. They are converted to real dates, and an invalid date becomes NULL instead of crashing the job.
- Product names contain double quotes (for example `8-1/2" x 11"`), so the CSV reader needs the escape character set to `"`.
- The phone column mixes numbers and text in Excel, so the sheet is read with every column as text.

**6. Suspicious profit values.**
A few rows look wrong, such as a profit of -16,438 on a line priced at 224.94. I left them as they are, since they come from the source, but they are worth a review by the business.

## Checks built into the pipeline

After the tables are written, the notebook checks that:
- no order lines were lost or duplicated by the joins
- there are no empty order dates or years
- every order matched a customer
- profit has no more than 2 decimal places
- the aggregate total equals the enriched orders total
- the aggregate table has no duplicate keys

If any check fails, the run stops with an error.

## Unit tests

The tests use small made-up DataFrames and cover:
- column renaming and the audit columns
- name cleaning: noisy names, legitimate names that must stay unchanged, and empty values
- date parsing, including that day and month are not swapped
- profit rounding
- removal of duplicate order lines
- joins: no extra rows when a product appears in several states, orphan products, and unknown customers
- aggregation, including floating-point rounding and reconciling to the enriched total
- each of the four SQL queries

## Reference results

To double-check the Spark output, I calculated the expected totals separately with pandas. After removing the duplicate line, the numbers should be:

| Year | Profit |
|---|---|
| 2014 | 39,197.81 |
| 2015 | 63,073.09 |
| 2016 | 65,073.23 |
| 2017 | 111,084.96 |

- Total profit: 278,429.09 across 9,993 order lines
- Rows in the aggregate table: 8,129

## Assumptions

- The year is taken from the order date, not the ship date.
- `Price` in the orders file is the line total (quantity times unit price), and it is not used in the profit calculation.
- Profit is taken directly from the orders file, not recalculated.
- An `Unknown` category is acceptable for products missing from the product file.

## What I would do with more time

- Partition `enriched_orders` by year and load orders with Auto Loader if the data grows large.
- Add a quarantine table for rows that fail checks, instead of only flagging them.
- Ask the business to confirm the extreme profit values and the handling of unknown products.
- Schedule the notebook as a Databricks Job with alerts on failure.
