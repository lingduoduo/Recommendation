WITH clicks AS (
    SELECT
        client_id,
        click_object_id AS item_id,
        click_details_caption AS title,
        TO_UNIX_TIMESTAMP(time_stamp, "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'") AS unix_timestamp,
        COUNT(*) AS clicks
    FROM
        onedata_us_east_1_shared_dit.nas_raw_lyric_search_dit.ml_search_with_click
    WHERE
        click_object_id IS NOT NULL 
        AND action = 'actions'
    GROUP BY
        client_id,
        click_object_id,
        click_details_caption,
        TO_UNIX_TIMESTAMP(time_stamp, "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
),

clicks_with_max AS (
    SELECT
        *,
        MAX(unix_timestamp) OVER () AS max_timestamp
    FROM clicks
)

SELECT
    client_id,
    item_id,
    title,
    SUM((1.0 / (1 + ((max_timestamp - unix_timestamp) / (24 * 60 * 60 * 100)))) * clicks) AS weighted_clicks
FROM
    clicks_with_max
GROUP BY
    client_id,
    item_id,
    title
ORDER BY
    client_id,
    weighted_clicks DESC;

--     The query enables cold-start recommendations by scoring and ranking items based on time-decayed clicks across all users, highlighting trending or popular items even when no personal history exists for the given client.
-- No reliance on individual history: It aggregates click data from all users, making it ideal for recommending items to new or low-activity clients.

-- Recency-aware ranking: By applying the decay factor 1 / (1 + delta), it emphasizes recent interactions, surfacing currently relevant items.

-- Popularity-based recommendation: Items are ranked by their decayed click scores, effectively prioritizing trending content likely to appeal broadly.