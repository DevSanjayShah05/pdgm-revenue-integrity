-- Revenue integrity summaries over the simulated claims (table `periods`, SYNTHETIC).

-- 1. Book-level summary
create or replace table summary as
select
    count(*)                                                        as periods,
    sum(payment_true)                                               as revenue_supported,
    sum(payment_billed)                                             as revenue_billed_paid,
    sum(payment_delta) filter (where payment_delta < 0)             as under_coded_dollars,
    sum(payment_delta) filter (where payment_delta > 0)             as over_coded_dollars,
    sum(payment_delta)                                              as net_dollars,
    avg(revenue_error::int)                                         as share_periods_with_revenue_error,
    avg(rtp::int)                                                   as rtp_rate,
    sum(payment_true) filter (where rtp)                            as dollars_delayed_by_rtp,
    avg((level_true <> level_billed)::int) filter (where not rtp)   as functional_level_mismatch_rate,
    avg((comorbidity_true <> comorbidity_billed)::int) filter (where not rtp) as comorbidity_mismatch_rate,
    avg((lupa_true <> lupa_billed)::int)                            as lupa_flip_rate
from periods;

-- 2. Leakage by clinical group (as truly documented)
create or replace table by_clinical_group as
select clinical_group_true as clinical_group, count(*) as periods,
       avg(payment_true) as avg_supported_payment,
       sum(payment_delta) filter (where payment_delta < 0) as under_coded_dollars,
       sum(payment_delta) filter (where payment_delta > 0) as over_coded_dollars,
       avg(revenue_error::int) as share_with_error
from periods group by 1 order by under_coded_dollars;

-- 3. Which PDGM component moved (paid claims with a revenue error)
create or replace table by_component as
select
    case when clinical_group_true <> clinical_group_billed then 'clinical group'
         when level_true <> level_billed then 'functional level'
         when comorbidity_true <> comorbidity_billed then 'comorbidity adjustment'
         when lupa_true <> lupa_billed then 'LUPA status'
         else 'other' end                     as component_changed,
    count(*) as periods, sum(payment_delta) as net_dollars, avg(abs(payment_delta)) as avg_abs_dollars
from periods where revenue_error group by 1 order by periods desc;

-- 4. Error-type co-occurrence on paid claims (exploded)
create or replace table by_error_tag as
with tags as (select period_id, payment_delta, revenue_error, unnest(string_split(errors, '|')) as tag
              from periods where errors <> '' and not rtp)
select tag, count(*) as periods_tagged, avg(revenue_error::int) as share_moving_payment,
       avg(payment_delta) as avg_delta_when_tagged
from tags group by 1 order by periods_tagged desc;
