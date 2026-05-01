-- ============================================================================
-- Clustering Output Validation Queries
-- Target: global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster
-- ============================================================================

-- 1. Record count and basic shape
SELECT
    COUNT(*) AS total_records,
    COUNT(DISTINCT commodity_cluster) AS n_commodity_clusters,
    COUNT(DISTINCT concat(commodity_cluster, '-', sub_commodity_cluster)) AS n_total_sub_clusters,
    COUNT(DISTINCT commodity) AS n_distinct_commodities,
    COUNT(DISTINCT sub_commodity) AS n_distinct_sub_commodities,
    MIN(clustered_at) AS run_timestamp,
    cluster_run_id
FROM global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster
GROUP BY cluster_run_id;

-- 2. Commodity cluster size distribution
SELECT
    commodity_cluster,
    COUNT(*) AS n_records,
    COUNT(DISTINCT commodity) AS n_commodities,
    COUNT(DISTINCT sub_commodity) AS n_sub_commodities,
    COUNT(DISTINCT parent_supplier_id) AS n_suppliers,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster
GROUP BY commodity_cluster
ORDER BY n_records DESC;

-- 3. Sub-commodity cluster distribution per commodity cluster
SELECT
    commodity_cluster,
    sub_commodity_cluster,
    COUNT(*) AS n_records,
    COUNT(DISTINCT sub_commodity) AS n_sub_commodities,
    COUNT(DISTINCT parent_supplier_id) AS n_suppliers
FROM global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster
GROUP BY commodity_cluster, sub_commodity_cluster
ORDER BY commodity_cluster, n_records DESC;

-- 4. Cluster balance — are clusters roughly even or heavily skewed?
SELECT
    'commodity' AS level,
    MIN(cluster_size) AS smallest,
    PERCENTILE_APPROX(cluster_size, 0.25) AS p25,
    PERCENTILE_APPROX(cluster_size, 0.50) AS median,
    PERCENTILE_APPROX(cluster_size, 0.75) AS p75,
    MAX(cluster_size) AS largest,
    ROUND(MAX(cluster_size) * 1.0 / MIN(cluster_size), 1) AS max_min_ratio,
    ROUND(STDDEV(cluster_size), 0) AS stddev
FROM (
    SELECT commodity_cluster, COUNT(*) AS cluster_size
    FROM global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster
    GROUP BY commodity_cluster
)
UNION ALL
SELECT
    'sub_commodity' AS level,
    MIN(cluster_size),
    PERCENTILE_APPROX(cluster_size, 0.25),
    PERCENTILE_APPROX(cluster_size, 0.50),
    PERCENTILE_APPROX(cluster_size, 0.75),
    MAX(cluster_size),
    ROUND(MAX(cluster_size) * 1.0 / MIN(cluster_size), 1),
    ROUND(STDDEV(cluster_size), 0)
FROM (
    SELECT commodity_cluster, sub_commodity_cluster, COUNT(*) AS cluster_size
    FROM global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster
    GROUP BY commodity_cluster, sub_commodity_cluster
);

-- 5. Dominant commodity per cluster — cluster purity
SELECT
    commodity_cluster,
    dominant_commodity,
    n_in_dominant,
    n_total,
    ROUND(n_in_dominant * 100.0 / n_total, 1) AS purity_pct,
    n_distinct_commodities
FROM (
    SELECT
        commodity_cluster,
        FIRST_VALUE(commodity) OVER (PARTITION BY commodity_cluster ORDER BY cnt DESC) AS dominant_commodity,
        FIRST_VALUE(cnt) OVER (PARTITION BY commodity_cluster ORDER BY cnt DESC) AS n_in_dominant,
        SUM(cnt) OVER (PARTITION BY commodity_cluster) AS n_total,
        COUNT(*) OVER (PARTITION BY commodity_cluster) AS n_distinct_commodities,
        ROW_NUMBER() OVER (PARTITION BY commodity_cluster ORDER BY cnt DESC) AS rn
    FROM (
        SELECT commodity_cluster, commodity, COUNT(*) AS cnt
        FROM global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster
        GROUP BY commodity_cluster, commodity
    )
)
WHERE rn = 1
ORDER BY purity_pct DESC;

-- 6. Top 3 commodities per cluster — what's in each group?
SELECT
    commodity_cluster,
    commodity,
    n_records,
    ROUND(n_records * 100.0 / cluster_total, 1) AS pct_of_cluster,
    rank_in_cluster
FROM (
    SELECT
        commodity_cluster,
        commodity,
        COUNT(*) AS n_records,
        SUM(COUNT(*)) OVER (PARTITION BY commodity_cluster) AS cluster_total,
        ROW_NUMBER() OVER (PARTITION BY commodity_cluster ORDER BY COUNT(*) DESC) AS rank_in_cluster
    FROM global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster
    GROUP BY commodity_cluster, commodity
)
WHERE rank_in_cluster <= 3
ORDER BY commodity_cluster, rank_in_cluster;

-- 7. Dominant sub-commodity per sub-cluster — sub-cluster purity
SELECT
    commodity_cluster,
    sub_commodity_cluster,
    dominant_sub_commodity,
    n_in_dominant,
    n_total,
    ROUND(n_in_dominant * 100.0 / n_total, 1) AS purity_pct
FROM (
    SELECT
        commodity_cluster,
        sub_commodity_cluster,
        FIRST_VALUE(sub_commodity) OVER (PARTITION BY commodity_cluster, sub_commodity_cluster ORDER BY cnt DESC) AS dominant_sub_commodity,
        FIRST_VALUE(cnt) OVER (PARTITION BY commodity_cluster, sub_commodity_cluster ORDER BY cnt DESC) AS n_in_dominant,
        SUM(cnt) OVER (PARTITION BY commodity_cluster, sub_commodity_cluster) AS n_total,
        ROW_NUMBER() OVER (PARTITION BY commodity_cluster, sub_commodity_cluster ORDER BY cnt DESC) AS rn
    FROM (
        SELECT commodity_cluster, sub_commodity_cluster, sub_commodity, COUNT(*) AS cnt
        FROM global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster
        GROUP BY commodity_cluster, sub_commodity_cluster, sub_commodity
    )
)
WHERE rn = 1
ORDER BY commodity_cluster, sub_commodity_cluster;

-- 8. Cross-cluster commodity leakage — commodities split across multiple clusters
SELECT
    commodity,
    COUNT(DISTINCT commodity_cluster) AS n_clusters,
    COUNT(*) AS total_records,
    COLLECT_SET(commodity_cluster) AS cluster_ids
FROM global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster
GROUP BY commodity
HAVING COUNT(DISTINCT commodity_cluster) > 1
ORDER BY n_clusters DESC, total_records DESC
LIMIT 20;

-- 9. Singleton check — clusters with very few records (potential noise)
SELECT
    commodity_cluster,
    sub_commodity_cluster,
    COUNT(*) AS n_records
FROM global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster
GROUP BY commodity_cluster, sub_commodity_cluster
HAVING COUNT(*) < 10
ORDER BY n_records;

-- 10. Supplier concentration — are any suppliers dominating a cluster?
SELECT
    commodity_cluster,
    sub_commodity_cluster,
    supplier_name,
    n_records,
    ROUND(n_records * 100.0 / cluster_total, 1) AS pct_of_cluster
FROM (
    SELECT
        commodity_cluster,
        sub_commodity_cluster,
        supplier_name,
        COUNT(*) AS n_records,
        SUM(COUNT(*)) OVER (PARTITION BY commodity_cluster, sub_commodity_cluster) AS cluster_total,
        ROW_NUMBER() OVER (PARTITION BY commodity_cluster, sub_commodity_cluster ORDER BY COUNT(*) DESC) AS rn
    FROM global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster
    GROUP BY commodity_cluster, sub_commodity_cluster, supplier_name
)
WHERE rn = 1 AND n_records * 100.0 / cluster_total > 50
ORDER BY pct_of_cluster DESC;

-- 11. Industry mix per commodity cluster
SELECT
    commodity_cluster,
    industry,
    COUNT(*) AS n_records,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY commodity_cluster), 1) AS pct
FROM global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster
GROUP BY commodity_cluster, industry
ORDER BY commodity_cluster, n_records DESC;

-- 12. Overall quality summary
SELECT
    COUNT(*) AS total_records,
    COUNT(DISTINCT commodity_cluster) AS commodity_clusters,
    COUNT(DISTINCT concat(commodity_cluster, '-', sub_commodity_cluster)) AS total_sub_clusters,
    ROUND(AVG(cluster_purity), 1) AS avg_commodity_purity_pct,
    ROUND(MIN(cluster_purity), 1) AS min_commodity_purity_pct,
    ROUND(MAX(cluster_purity), 1) AS max_commodity_purity_pct
FROM global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster
LEFT JOIN (
    SELECT
        commodity_cluster,
        MAX(cnt) * 100.0 / SUM(cnt) AS cluster_purity
    FROM (
        SELECT commodity_cluster, commodity, COUNT(*) AS cnt
        FROM global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster
        GROUP BY commodity_cluster, commodity
    )
    GROUP BY commodity_cluster
) purity USING (commodity_cluster);
