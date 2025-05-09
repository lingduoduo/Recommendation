with clicks as (
    SELECT
        client_id,
        click_object_id AS item_id,
        click_details_caption AS title,
        TO_UNIX_TIMESTAMP(time_stamp, "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'") AS unix_timestamp,
        count(*) AS clicks
    FROM
        onedata_us_east_1_shared_dit.nas_raw_lyric_search_dit.ml_search_with_click
    WHERE
        click_object_id IS NOT NULL and action = 'actions'
    GROUP BY
        client_id,
        click_object_id,
        click_details_caption,
        TO_UNIX_TIMESTAMP(time_stamp, "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'") 
),

max_ts as (
    select max(unix_timestamp) as max_timestamp from clicks
)

select res.client_id, res.item_id, res.title, sum((1/(1+res.delta)) * res.clicks) 
from (
    select 
        clicks.*, 
        max_ts.max_timestamp, 
        (max_ts.max_timestamp - clicks.unix_timestamp) / (24 * 60 * 60 * 100) as delta
    from clicks 
    join max_ts
) res
group by 1, 2, 3
order by 1, 4 desc
