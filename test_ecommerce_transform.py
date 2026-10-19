"""
Unit tests for ecommerce_transforms.py
"""
import datetime as dt
import os

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import ecommerce_transforms as T


@pytest.fixture(scope="session")
def spark():
    on_databricks = "DATABRICKS_RUNTIME_VERSION" in os.environ or "SPARK_REMOTE" in os.environ
    builder = SparkSession.builder
    if not on_databricks:
        builder = (builder.master("local[2]").appName("ecommerce-tests")
                   .config("spark.sql.shuffle.partitions", "2"))
    s = builder.getOrCreate()
    try:
        s.conf.set("spark.sql.session.timeZone", "UTC")
    except Exception:
        pass
    yield s


#fixtures
RAW_ORDER_SCHEMA = ("row_id long, order_id string, order_date string, ship_date string, "
                    "ship_mode string, customer_id string, product_id string, quantity long, "
                    "price double, discount double, profit double")
RAW_CUSTOMER_SCHEMA = ("customer_id string, customer_name string, segment string, country string, "
                       "city string, state string, postal_code long, region string, phone string")
RAW_PRODUCT_SCHEMA = ("product_id string, category string, sub_category string, "
                      "product_name string, state string, price_per_product double")


def orders(spark, rows):
    return spark.createDataFrame(rows, RAW_ORDER_SCHEMA)


def customers(spark, rows):
    return spark.createDataFrame(rows, RAW_CUSTOMER_SCHEMA)


def products(spark, rows):
    return spark.createDataFrame(rows, RAW_PRODUCT_SCHEMA)


def order_row(row_id=1, order_id="CA-2016-1", order_date="21/8/2016", cust="AA-1", prod="P1",
              qty=1, price=10.0, disc=0.0, profit=1.0):
    return (row_id, order_id, order_date, "25/8/2016", "Standard Class", cust, prod, qty, price, disc, profit)


def cust_row(cid="AA-1", name="Alice Adams", country="United States"):
    return (cid, name, "Consumer", country, "Austin", "Texas", 1001, "South", "555")


def prod_row(pid="P1", cat="Furniture", sub="Chairs", name="Chair", state="Texas", price=9.999):
    return (pid, cat, sub, name, state, price)


# ---------------------------------------------------------------- helpers
class TestHelpers:
    @pytest.mark.parametrize("raw,expected", [
        ("Order ID", "order_id"),
        ("Sub-Category", "sub_category"),
        ("Price per product", "price_per_product"),
        ("  Postal Code ", "postal_code"),
        ("email", "email"),
    ])
    def test_to_snake_case(self, raw, expected):
        assert T.to_snake_case(raw) == expected

    def test_standardise_columns(self, spark):
        df = spark.createDataFrame([(1, "x")], "`Row ID` int, `Sub-Category` string")
        assert T.standardise_columns(df).columns == ["row_id", "sub_category"]

    def test_add_ingest_metadata(self, spark):
        df = T.add_ingest_metadata(spark.createDataFrame([(1,)], "a int"), "f.csv")
        assert {"_source_file", "_ingested_at"} <= set(df.columns)
        assert df.first()["_source_file"] == "f.csv"


#name cleaning
class TestCleanPersonName:
    @pytest.mark.parametrize("raw,expected", [
        ("Tam&^*ara Willing___)ham", "Tamara Willingham"),   # junk inside words
        ("Ad.       ..am Hart", "Adam Hart"),                 # junk + spaces inside a word
        ("B         ecky Martin", "Becky Martin"),            # spaces only inside a word
        ("Gary567 Hansen", "Gary Hansen"),                    # digits at word boundary
        ("_Mike Vitt 12313orini", "Mike Vittorini"),          # leading junk + mid-word run
        ("&&Tracy Blumstein", "Tracy Blumstein"),             # leading junk
        ("Neil Knudson%^&*(", "Neil Knudson"),                # trailing junk
        ("Jason Fortune-", "Jason Fortune"),                  # dangling hyphen
        ("Paul  4567Van Hugh", "Paul Van Hugh"),              # junk run before capital = word break
        ("   Kristi;'[]na Nunn", "Kristina Nunn"),            # leading spaces + apostrophe junk
        ("Shahi  Hopkins", "Shahi Hopkins"),                  # double space between words
    ])
    def test_removes_noise(self, spark, raw, expected):
        df = spark.createDataFrame([(raw,)], "n string").select(T.clean_person_name(F.col("n")).alias("c"))
        assert df.first()["c"] == expected

    @pytest.mark.parametrize("legit", ["Mary O'Rourke", "Peter Bühler", "Anna Bell-Smith", "Neil Französisch"])
    def test_keeps_legitimate_names_unchanged(self, spark, legit):
        df = spark.createDataFrame([(legit,)], "n string").select(T.clean_person_name(F.col("n")).alias("c"))
        assert df.first()["c"] == legit

    @pytest.mark.parametrize("raw", [None, "", "   ", "123 !!!"])
    def test_null_or_empty_becomes_null(self, spark, raw):
        df = spark.createDataFrame([(raw,)], "n string").select(T.clean_person_name(F.col("n")).alias("c"))
        assert df.first()["c"] is None


#enriched customers / products
class TestEnrichCustomers:
    def test_cleans_name_and_types(self, spark):
        raw = customers(spark, [("AA-1", "Al&^ice Adams", " Consumer ", " United States ", "Austin", "Texas", 1001, "South", "555")])
        r = T.enrich_customers(raw).first()
        assert r["customer_name"] == "Alice Adams"
        assert r["country"] == "United States"      
        assert r["segment"] == "Consumer"
        assert r["postal_code"] == "1001"            
        assert isinstance(r["phone"], str)

    def test_null_name_preserved(self, spark):
        out = T.enrich_customers(customers(spark, [cust_row(name=None)]))
        assert out.first()["customer_name"] is None
        assert out.count() == 1

    def test_deduplicates_on_customer_id(self, spark):
        out = T.enrich_customers(customers(spark, [cust_row(), cust_row()]))
        assert out.count() == 1


class TestEnrichProducts:
    def test_trims_and_rounds(self, spark):
        raw = products(spark, [prod_row(name="  Chair  ", price=81.882)])
        r = T.enrich_products(raw).first()
        assert r["product_name"] == "Chair"
        assert r["price_per_product"] == 81.88

    def test_keeps_state_grain_but_drops_exact_duplicates(self, spark):
        raw = products(spark, [prod_row(state="Texas"), prod_row(state="Texas"), prod_row(state="Ohio")])
        assert T.enrich_products(raw).count() == 2

    def test_product_dimension_one_row_per_id(self, spark):
        raw = products(spark, [prod_row(name="Chair A", state="Texas"),
                               prod_row(name="Chair B", state="Ohio"),
                               prod_row(pid="P2", cat="Technology", sub="Phones", state="Ohio")])
        dim = T.product_dimension(T.enrich_products(raw))
        assert dim.count() == 2
        assert dim.filter("product_id = 'P1'").count() == 1


#clean orders
class TestCleanOrders:
    def test_date_parsing_and_year(self, spark):
        r = T.clean_orders(orders(spark, [order_row(order_date="21/8/2016")])).first()
        assert r["order_date"] == dt.date(2016, 8, 21)
        assert r["ship_date"] == dt.date(2016, 8, 25)
        assert r["order_year"] == 2016

    def test_day_month_not_swapped(self, spark):
        r = T.clean_orders(orders(spark, [order_row(order_date="3/10/2014")])).first()
        assert r["order_date"] == dt.date(2014, 10, 3)

    @pytest.mark.parametrize("raw,expected", [
        (63.694, 63.69), (5.646, 5.65), (-14.924, -14.92), (-3.0, -3.0), (0.0, 0.0), (12452.0, 12452.0),
    ])
    def test_profit_rounded_to_2dp(self, spark, raw, expected):
        assert T.clean_orders(orders(spark, [order_row(profit=raw)])).first()["profit"] == expected

    def test_invalid_date_becomes_null_not_error(self, spark):
        r = T.clean_orders(orders(spark, [order_row(order_date="31/02/2016")])).first()
        assert r["order_date"] is None and r["order_year"] is None

    def test_exact_duplicate_lines_removed_ignoring_row_id(self, spark):
        rows = [order_row(row_id=1), order_row(row_id=2), order_row(row_id=3, order_id="CA-2016-2")]
        assert T.clean_orders(orders(spark, rows)).count() == 2

    def test_dedupe_can_be_disabled(self, spark):
        rows = [order_row(row_id=1), order_row(row_id=2)]
        assert T.clean_orders(orders(spark, rows), dedupe=False).count() == 2

    def test_different_lines_same_order_are_kept(self, spark):
        rows = [order_row(row_id=1, prod="P1"), order_row(row_id=2, prod="P2")]
        assert T.clean_orders(orders(spark, rows)).count() == 2


#enriched orders
class TestBuildEnrichedOrders:
    def _build(self, spark, o, c, p, **kw):
        return T.build_enriched_orders(orders(spark, o),
                                       T.enrich_customers(customers(spark, c)),
                                       T.enrich_products(products(spark, p)), **kw)

    def test_has_required_columns(self, spark):
        out = self._build(spark, [order_row()], [cust_row()], [prod_row()])
        for col in ("order_id", "order_date", "profit", "customer_name", "customer_country", "category", "sub_category"):
            assert col in out.columns

    def test_enrichment_values(self, spark):
        r = self._build(spark, [order_row(profit=63.694)], [cust_row(name="Al&ice Adams")], [prod_row()]).first()
        assert (r["customer_name"], r["customer_country"]) == ("Alice Adams", "United States")
        assert (r["category"], r["sub_category"]) == ("Furniture", "Chairs")
        assert r["profit"] == 63.69
        assert r["product_found"] is True

    def test_no_row_fan_out_when_product_in_many_states(self, spark):
        prods = [prod_row(state=s, name=f"Chair {s}") for s in ("Texas", "Ohio", "Utah")]
        out = self._build(spark, [order_row()], [cust_row()], prods)
        assert out.count() == 1                                   
        assert out.first()["profit"] == 1.0

    def test_orphan_product_kept_as_unknown(self, spark):
        r = self._build(spark, [order_row(prod="NOPE", profit=7.0)], [cust_row()], [prod_row()]).first()
        assert (r["category"], r["sub_category"]) == ("Unknown", "Unknown")
        assert r["product_found"] is False
        assert r["profit"] == 7.0                                 

    def test_orphan_customer_kept_with_null_name(self, spark):
        out = self._build(spark, [order_row(cust="ZZ-9")], [cust_row()], [prod_row()])
        assert out.count() == 1
        assert out.first()["customer_name"] is None

    def test_row_count_equals_deduped_orders(self, spark):
        o = [order_row(row_id=i, order_id=f"O{i}") for i in range(1, 6)] + [order_row(row_id=9, order_id="O1")]
        out = self._build(spark, o, [cust_row()], [prod_row()])
        assert out.count() == 5                                   


#aggregate
class TestProfitAggregate:
    def _enriched(self, spark, rows):
        o = [order_row(row_id=i + 1, order_id=f"O{i}", **r) for i, r in enumerate(rows)]
        return T.build_enriched_orders(
            orders(spark, o),
            T.enrich_customers(customers(spark, [cust_row("AA-1", "Alice Adams"), cust_row("BB-2", "Bob Brown")])),
            T.enrich_products(products(spark, [prod_row("P1", "Furniture", "Chairs"),
                                               prod_row("P2", "Technology", "Phones")])))

    def test_groups_and_sums(self, spark):
        enr = self._enriched(spark, [
            dict(cust="AA-1", prod="P1", order_date="1/1/2016", profit=10.0),
            dict(cust="AA-1", prod="P1", order_date="2/2/2016", profit=5.5),     # same grain
            dict(cust="AA-1", prod="P1", order_date="1/1/2017", profit=1.0),     # different year
            dict(cust="BB-2", prod="P2", order_date="1/1/2016", profit=-3.0),    # different everything
        ])
        agg = {(r["order_year"], r["category"], r["customer_id"]): r["total_profit"]
               for r in T.build_profit_aggregate(enr).collect()}
        assert agg == {(2016, "Furniture", "AA-1"): 15.5,
                       (2017, "Furniture", "AA-1"): 1.0,
                       (2016, "Technology", "BB-2"): -3.0}

    def test_floating_point_sums_are_rounded(self, spark):
        enr = self._enriched(spark, [dict(profit=0.1), dict(profit=0.2)])
        assert T.build_profit_aggregate(enr).first()["total_profit"] == 0.3

    def test_grain_has_no_duplicate_keys(self, spark):
        enr = self._enriched(spark, [dict(profit=1.0), dict(profit=2.0), dict(prod="P2", profit=3.0)])
        agg = T.build_profit_aggregate(enr)
        assert agg.count() == agg.dropDuplicates(T.AGG_KEYS).count()

    def test_reconciles_to_enriched_total(self, spark):
        enr = self._enriched(spark, [dict(profit=1.11), dict(profit=2.22, prod="P2"), dict(profit=-0.33, cust="BB-2"),
                                     dict(profit=4.0, prod="UNKNOWN")])
        total_enriched = enr.agg(F.round(F.sum("profit"), 2)).first()[0]
        total_agg = T.build_profit_aggregate(enr).agg(F.round(F.sum("total_profit"), 2)).first()[0]
        assert total_agg == total_enriched == 7.0


# SQL outputs
class TestSqlOutputs:
    @pytest.fixture()
    def agg_view(self, spark):
        rows = [(2016, "Furniture", "Chairs", "AA-1", "Alice Adams", 10.0, 1),
                (2016, "Technology", "Phones", "AA-1", "Alice Adams", 5.0, 1),
                (2017, "Furniture", "Chairs", "AA-1", "Alice Adams", 1.0, 1),
                (2016, "Furniture", "Chairs", "BB-2", "Bob Brown", -3.0, 1)]
        spark.createDataFrame(rows, "order_year int, category string, sub_category string, customer_id string, "
                                    "customer_name string, total_profit double, order_lines long") \
             .createOrReplaceTempView("agg_test")
        yield "agg_test"
        spark.catalog.dropTempView("agg_test")

    def run(self, spark, name, view):
        return [r.asDict() for r in spark.sql(T.SQL_QUERIES[name].format(t=view)).collect()]

    def test_by_year(self, spark, agg_view):
        assert self.run(spark, "profit_by_year", agg_view) == [
            {"order_year": 2016, "profit": 12.0}, {"order_year": 2017, "profit": 1.0}]

    def test_by_year_category(self, spark, agg_view):
        res = {(r["order_year"], r["category"]): r["profit"] for r in self.run(spark, "profit_by_year_category", agg_view)}
        assert res == {(2016, "Furniture"): 7.0, (2016, "Technology"): 5.0, (2017, "Furniture"): 1.0}

    def test_by_customer(self, spark, agg_view):
        res = {r["customer_id"]: r["profit"] for r in self.run(spark, "profit_by_customer", agg_view)}
        assert res == {"AA-1": 16.0, "BB-2": -3.0}

    def test_by_customer_year(self, spark, agg_view):
        res = {(r["customer_id"], r["order_year"]): r["profit"] for r in self.run(spark, "profit_by_customer_year", agg_view)}
        assert res == {("AA-1", 2016): 15.0, ("AA-1", 2017): 1.0, ("BB-2", 2016): -3.0}

    def test_every_query_sums_to_the_same_total(self, spark, agg_view):
        totals = {n: round(sum(r["profit"] for r in self.run(spark, n, agg_view)), 2) for n in T.SQL_QUERIES}
        assert set(totals.values()) == {13.0}
