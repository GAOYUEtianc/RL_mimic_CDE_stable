"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MIMIC-III Sepsis Cohort Extraction.

This file is sourced and modified from: https://github.com/matthieukomorowski/AI_Clinician using bigquery to fetch data
"""

import argparse
import os
import pandas as pd
from google.cloud import bigquery


# Initialize BigQuery Client
client = bigquery.Client()
dataset_id = "mimic3_v1_4"

# Path for processed data storage
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
exportdir = os.path.join(repo_root, 'data/sepsis_mimiciii')

if not os.path.exists(exportdir):
    os.makedirs(exportdir)

# Function to query BigQuery and save results as CSV
def extract_and_save(query, filename):
    query_job = client.query(query)
    df = query_job.to_dataframe()
    file_path = os.path.join(exportdir, filename)
    df.to_csv(file_path, index=False)
    print(f"Saved {filename}")
    
# Extraction of sub-tables
# There are 43 tables in the Mimic III database. 
# 26 unique tables; the other 17 are partitions of chartevents that are not to be queried directly 
# See: https://mit-lcp.github.io/mimic-schema-spy/
# We create 15 sub-tables when extracting from the database

# From each table we extract subject ID, admission ID, ICU stay ID 
# and relevant times to assist in joining these tables
# All other specific information extracted will be documented before each section of the following code.


# NOTE: The next three tables are built to help identify when a patient may be 
# considered to be septic, using the Sepsis 3 criteria

# 1. culture
# These correspond to blood/urine/CSF/sputum cultures etc
# There are 18 chartevent tables in the Mimic III database, one unsubscripted and 
# the others subscripted from 1 to 17. We use the unsubscripted one to create the 
# culture subtable. The remaining 17 are just partitions and should not be directly queried.
# The labels corresponding to the 51 itemids in the query below are:
"""
 Itemid | Label
-----------------------------------------------------
    938 | blood cultures
    941 | urine culture
    942 | BLOOD CULTURES
   2929 | sputum culture
   3333 | Blood Cultures
   4855 | Urine culture
   6035 | Urinalysis sent
   6043 | surface cultures
  70006 | ANORECTAL/VAGINAL CULTURE
  70011 | BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)
  70012 | BLOOD CULTURE
  70013 | FLUID RECEIVED IN BLOOD CULTURE BOTTLES
  70014 | BLOOD CULTURE - NEONATE
  70016 | BLOOD CULTURE (POST-MORTEM)
  70024 | VIRAL CULTURE: R/O CYTOMEGALOVIRUS
  70037 | FOOT CULTURE
  70041 | VIRAL CULTURE:R/O HERPES SIMPLEX VIRUS
  70055 | POSTMORTEM CULTURE
  70057 | Rapid Respiratory Viral Screen & Culture
  70060 | Stem Cell - Blood Culture
  70063 | STERILITY CULTURE
  70075 | THROAT CULTURE
  70083 | VARICELLA-ZOSTER CULTURE
  80220 | AFB GROWN IN CULTURE; ADDITIONAL INFORMATION TO FOLLOW
 225401 | Blood Cultured
 225437 | CSF Culture
 225444 | Pan Culture
 225451 | Sputum Culture
 225454 | Urine Culture
 225722 | Arterial Line Tip Cultured
 225723 | CCO PAC Line Tip Cultured
 225724 | Cordis/Introducer Line Tip Cultured
 225725 | Dialysis Catheter Tip Cultured
 225726 | Tunneled (Hickman) Line Tip Cultured
 225727 | IABP Line Tip Cultured
 225728 | Midline Tip Cultured
 225729 | Multi Lumen Line Tip Cultured
 225730 | PA Catheter Line Tip Cultured
 225731 | Pheresis Catheter Line Tip Cultured
 225732 | PICC Line Tip Cultured
 225733 | Indwelling Port (PortaCath) Line Tip Cultured
 225734 | Presep Catheter Line Tip Cultured
 225735 | Trauma Line Tip Cultured
 225736 | Triple Introducer Line Tip Cultured
 225768 | Sheath Line Tip Cultured
 225814 | Stool Culture
 225816 | Wound Culture
 225817 | BAL Fluid Culture
 225818 | Pleural Fluid Culture
 226131 | ICP Line Tip Cultured
 227726 | AVA Line Tip Cultured
"""
query_culture = f"""
SELECT subject_id, hadm_id, icustay_id, 
       UNIX_SECONDS(TIMESTAMP(charttime)) AS charttime, 
       itemid
FROM `mimic3_v1_4.CHARTEVENTS`
WHERE itemid IN (6035, 3333, 938, 941, 942, 4855, 6043, 2929, 225401, 225437, 225444, 225451, 225454, 225814,
  225816, 225817, 225818, 225722, 225723, 225724, 225725, 225726, 225727, 225728, 225729, 225730, 225731,
  225732, 225733, 227726, 70006, 70011, 70012, 70013, 70014, 70016, 70024, 70037, 70041, 225734, 225735,
  225736, 225768, 70055, 70057, 70060, 70063, 70075, 70083, 226131, 80220)
ORDER BY subject_id, hadm_id, charttime
"""

extract_and_save(query_culture, 'culture.csv')


# 2. microbio (Microbiologyevents)
query_microbio = """
SELECT subject_id, hadm_id, 
       UNIX_SECONDS(TIMESTAMP(charttime)) AS charttime, 
       UNIX_SECONDS(TIMESTAMP(chartdate)) AS chartdate
FROM `mimic3_v1_4.MICROBIOLOGYEVENTS`
"""

extract_and_save(query_microbio, 'microbio.csv')


# 3. abx (Antibiotics administration)
# gsn/GSN: Generic Sequence Number. This number provides a representation of the drug in various coding systems. 
# GSN is First DataBank's classification system. These are 6 digit codes for various drugs.
# ???  The codes here correspond to various antibiotics as sepsis onset is detected by administration of antibiotcs ???
query_abx = """
SELECT hadm_id, icustay_id, 
       UNIX_SECONDS(TIMESTAMP(startdate)) AS startdate, 
       UNIX_SECONDS(TIMESTAMP(enddate)) AS enddate
FROM `mimic3_v1_4.PRESCRIPTIONS`
WHERE gsn IN ('002542','002543','007371','008873','008877','008879','008880','008935','008941',
  '008942','008943','008944','008983','008984','008990','008991','008992','008995','008996',
  '008998','009043','009046','009065','009066','009136','009137','009162','009164','009165',
  '009171','009182','009189','009213','009214','009218','009219','009221','009226','009227',
  '009235','009242','009263','009273','009284','009298','009299','009310','009322','009323',
  '009326','009327','009339','009346','009351','009354','009362','009394','009395','009396',
  '009509','009510','009511','009544','009585','009591','009592','009630','013023','013645',
  '013723','013724','013725','014182','014500','015979','016368','016373','016408','016931',
  '016932','016949','018636','018637','018766','019283','021187','021205','021735','021871',
  '023372','023989','024095','024194','024668','025080','026721','027252','027465','027470',
  '029325','029927','029928','037042','039551','039806','040819','041798','043350','043879',
  '044143','045131','045132','046771','047797','048077','048262','048266','048292','049835',
  '050442','050443','051932','052050','060365','066295','067471')
ORDER BY hadm_id, icustay_id
"""

extract_and_save(query_abx, 'abx.csv')

# 4. demog (Patient demographics)
# See https://github.com/MIT-LCP/mimic-code/blob/master/concepts/comorbidity/elixhauser-quan.sql
# This code calculates the Elixhauser comorbidities as defined in Quan et. al 2009
# This outputs a materialized view (table) with 58976 rows and 31 columns. The first column is 'hadm_id' and the 
# rest of the columns are as given below (Each entry is either 0 or 1):
# 2. 'congestive_heart_failure', 
# 3. 'cardiac_arrhythmias',
# 4. 'valvular_disease',
# 5. 'pulmonary_circulation', 
# 6. 'peripheral_vascular',
# 7. 'hypertension', 
# 8. 'paralysis', 
# 9. 'other_neurological'
# 10.'chronic_pulmonary',
# 11. 'diabetes_uncomplicated', 
# 12. 'diabetes_complicated', 
# 13. 'hypothyroidism',
# 14. 'renal_failure', 
# 15. 'liver_disease', 
# 16. 'peptic_ulcer', 
# 17. 'aids', 
# 18. 'lymphoma',
# 19. 'metastatic_cancer', 
# 20. 'solid_tumor', 
# 21. 'rheumatoid_arthritis',
# 22. 'coagulopathy', 
# 23. 'obesity', 
# 24. 'weight_loss', 
# 25. 'fluid_electrolyte',
# 26. 'blood_loss_anemia', 
# 27. 'deficiency_anemias', 
# 28. 'alcohol_abuse',
# 29. 'drug_abuse', 
# 30. 'psychoses', 
# 31. 'depression'
def save_elixhauser_to_bq():
    client = bigquery.Client()
    
    # Define the destination table
    table_id = "ELIXHAUSER_QUAN"
    destination_table = f"{client.project}.public.{table_id}"
    
    # Define the query
    query = """
    -- This code calculates the Elixhauser comorbidities as defined in Quan et. al 2009:
-- Quan, Hude, et al. "Coding algorithms for defining comorbidities in
-- ICD-9-CM and ICD-10 administrative data." Medical care (2005): 1130-1139.
--  https://www.ncbi.nlm.nih.gov/pubmed/16224307

-- Quan defined an "Enhanced ICD-9" coding scheme for deriving Elixhauser
-- comorbidities from ICD-9 billing codes. This script implements that calculation.

-- The logic of the code is roughly that, if the comorbidity lists a length 3
-- ICD-9 code (e.g. 585), then we only require a match on the first 3 characters.

-- This code derives each comorbidity as follows:
--  1) ICD9_CODE is directly compared to 5 character codes
--  2) The first 4 characters of ICD9_CODE are compared to 4 character codes
--  3) The first 3 characters of ICD9_CODE are compared to 3 character codes
with eliflg as
(
select hadm_id, seq_num, icd9_code
, CASE
  when icd9_code in ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493') then 1
  when SUBSTR(icd9_code, 1, 4) in ('4254','4255','4257','4258','4259') then 1
  when SUBSTR(icd9_code, 1, 3) in ('428') then 1
  else 0 end as chf       /* Congestive heart failure */

, CASE
  when icd9_code in ('42613','42610','42612','99601','99604') then 1
  when SUBSTR(icd9_code, 1, 4) in ('4260','4267','4269','4270','4271','4272','4273','4274','4276','4278','4279','7850','V450','V533') then 1
  else 0 end as arrhy

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('0932','7463','7464','7465','7466','V422','V433') then 1
  when SUBSTR(icd9_code, 1, 3) in ('394','395','396','397','424') then 1
  else 0 end as valve     /* Valvular disease */

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('4150','4151','4170','4178','4179') then 1
  when SUBSTR(icd9_code, 1, 3) in ('416') then 1
  else 0 end as pulmcirc  /* Pulmonary circulation disorder */

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('0930','4373','4431','4432','4438','4439','4471','5571','5579','V434') then 1
  when SUBSTR(icd9_code, 1, 3) in ('440','441') then 1
  else 0 end as perivasc  /* Peripheral vascular disorder */

, CASE
  when SUBSTR(icd9_code, 1, 3) in ('401') then 1
  else 0 end as htn       /* Hypertension, uncomplicated */

, CASE
  when SUBSTR(icd9_code, 1, 3) in ('402','403','404','405') then 1
  else 0 end as htncx     /* Hypertension, complicated */

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('3341','3440','3441','3442','3443','3444','3445','3446','3449') then 1
  when SUBSTR(icd9_code, 1, 3) in ('342','343') then 1
  else 0 end as para      /* Paralysis */

, CASE
  when icd9_code in ('33392') then 1
  when SUBSTR(icd9_code, 1, 4) in ('3319','3320','3321','3334','3335','3362','3481','3483','7803','7843') then 1
  when SUBSTR(icd9_code, 1, 3) in ('334','335','340','341','345') then 1
  else 0 end as neuro     /* Other neurological */

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('4168','4169','5064','5081','5088') then 1
  when SUBSTR(icd9_code, 1, 3) in ('490','491','492','493','494','495','496','500','501','502','503','504','505') then 1
  else 0 end as chrnlung  /* Chronic pulmonary disease */

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('2500','2501','2502','2503') then 1
  else 0 end as dm        /* Diabetes w/o chronic complications*/

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('2504','2505','2506','2507','2508','2509') then 1
  else 0 end as dmcx      /* Diabetes w/ chronic complications */

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('2409','2461','2468') then 1
  when SUBSTR(icd9_code, 1, 3) in ('243','244') then 1
  else 0 end as hypothy   /* Hypothyroidism */

, CASE
  when icd9_code in ('40301','40311','40391','40402','40403','40412','40413','40492','40493') then 1
  when SUBSTR(icd9_code, 1, 4) in ('5880','V420','V451') then 1
  when SUBSTR(icd9_code, 1, 3) in ('585','586','V56') then 1
  else 0 end as renlfail  /* Renal failure */

, CASE
  when icd9_code in ('07022','07023','07032','07033','07044','07054') then 1
  when SUBSTR(icd9_code, 1, 4) in ('0706','0709','4560','4561','4562','5722','5723','5724','5728','5733','5734','5738','5739','V427') then 1
  when SUBSTR(icd9_code, 1, 3) in ('570','571') then 1
  else 0 end as liver     /* Liver disease */

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('5317','5319','5327','5329','5337','5339','5347','5349') then 1
  else 0 end as ulcer     /* Chronic Peptic ulcer disease (includes bleeding only if obstruction is also present) */

, CASE
  when SUBSTR(icd9_code, 1, 3) in ('042','043','044') then 1
  else 0 end as aids      /* HIV and AIDS */

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('2030','2386') then 1
  when SUBSTR(icd9_code, 1, 3) in ('200','201','202') then 1
  else 0 end as lymph     /* Lymphoma */

, CASE
  when SUBSTR(icd9_code, 1, 3) in ('196','197','198','199') then 1
  else 0 end as mets      /* Metastatic cancer */

, CASE
  when SUBSTR(icd9_code, 1, 3) in
  (
     '140','141','142','143','144','145','146','147','148','149','150','151','152'
    ,'153','154','155','156','157','158','159','160','161','162','163','164','165'
    ,'166','167','168','169','170','171','172','174','175','176','177','178','179'
    ,'180','181','182','183','184','185','186','187','188','189','190','191','192'
    ,'193','194','195'
  ) then 1
  else 0 end as tumor     /* Solid tumor without metastasis */

, CASE
  when icd9_code in ('72889','72930') then 1
  when SUBSTR(icd9_code, 1, 4) in ('7010','7100','7101','7102','7103','7104','7108','7109','7112','7193','7285') then 1
  when SUBSTR(icd9_code, 1, 3) in ('446','714','720','725') then 1
  else 0 end as arth              /* Rheumatoid arthritis/collagen vascular diseases */

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('2871','2873','2874','2875') then 1
  when SUBSTR(icd9_code, 1, 3) in ('286') then 1
  else 0 end as coag      /* Coagulation deficiency */

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('2780') then 1
  else 0 end as obese     /* Obesity      */

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('7832','7994') then 1
  when SUBSTR(icd9_code, 1, 3) in ('260','261','262','263') then 1
  else 0 end as wghtloss  /* Weight loss */

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('2536') then 1
  when SUBSTR(icd9_code, 1, 3) in ('276') then 1
  else 0 end as lytes     /* Fluid and electrolyte disorders */

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('2800') then 1
  else 0 end as bldloss   /* Blood loss anemia */

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('2801','2808','2809') then 1
  when SUBSTR(icd9_code, 1, 3) in ('281') then 1
  else 0 end as anemdef  /* Deficiency anemias */

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('2652','2911','2912','2913','2915','2918','2919','3030','3039','3050','3575','4255','5353','5710','5711','5712','5713','V113') then 1
  when SUBSTR(icd9_code, 1, 3) in ('980') then 1
  else 0 end as alcohol /* Alcohol abuse */

, CASE
  when icd9_code in ('V6542') then 1
  when SUBSTR(icd9_code, 1, 4) in ('3052','3053','3054','3055','3056','3057','3058','3059') then 1
  when SUBSTR(icd9_code, 1, 3) in ('292','304') then 1
  else 0 end as drug /* Drug abuse */

, CASE
  when icd9_code in ('29604','29614','29644','29654') then 1
  when SUBSTR(icd9_code, 1, 4) in ('2938') then 1
  when SUBSTR(icd9_code, 1, 3) in ('295','297','298') then 1
  else 0 end as psych /* Psychoses */

, CASE
  when SUBSTR(icd9_code, 1, 4) in ('2962','2963','2965','3004') then 1
  when SUBSTR(icd9_code, 1, 3) in ('309','311') then 1
  else 0 end as depress  /* Depression */
from `physionet-data.mimiciii_clinical.diagnoses_icd` icd
where seq_num != 1 -- we do not include the primary icd-9 code
)
-- collapse the icd9_code specific flags into hadm_id specific flags
-- this groups comorbidities together for a single patient admission
, eligrp as
(
  select hadm_id
  , max(chf) as chf
  , max(arrhy) as arrhy
  , max(valve) as valve
  , max(pulmcirc) as pulmcirc
  , max(perivasc) as perivasc
  , max(htn) as htn
  , max(htncx) as htncx
  , max(para) as para
  , max(neuro) as neuro
  , max(chrnlung) as chrnlung
  , max(dm) as dm
  , max(dmcx) as dmcx
  , max(hypothy) as hypothy
  , max(renlfail) as renlfail
  , max(liver) as liver
  , max(ulcer) as ulcer
  , max(aids) as aids
  , max(lymph) as lymph
  , max(mets) as mets
  , max(tumor) as tumor
  , max(arth) as arth
  , max(coag) as coag
  , max(obese) as obese
  , max(wghtloss) as wghtloss
  , max(lytes) as lytes
  , max(bldloss) as bldloss
  , max(anemdef) as anemdef
  , max(alcohol) as alcohol
  , max(drug) as drug
  , max(psych) as psych
  , max(depress) as depress
from eliflg
group by hadm_id
)
-- now merge these flags together to define elixhauser
-- most are straightforward.. but hypertension flags are a bit more complicated


select adm.hadm_id
, chf as congestive_heart_failure
, arrhy as cardiac_arrhythmias
, valve as valvular_disease
, pulmcirc as pulmonary_circulation
, perivasc as peripheral_vascular
-- we combine "htn" and "htncx" into "HYPERTENSION"
, case
    when htn = 1 then 1
    when htncx = 1 then 1
  else 0 end as hypertension
, para as paralysis
, neuro as other_neurological
, chrnlung as chronic_pulmonary
-- only the more severe comorbidity (complicated diabetes) is kept
, case
    when dmcx = 1 then 0
    when dm = 1 then 1
  else 0 end as diabetes_uncomplicated
, dmcx as diabetes_complicated
, hypothy as hypothyroidism
, renlfail as renal_failure
, liver as liver_disease
, ulcer as peptic_ulcer
, aids as aids
, lymph as lymphoma
, mets as metastatic_cancer
-- only the more severe comorbidity (metastatic cancer) is kept
, case
    when mets = 1 then 0
    when tumor = 1 then 1
  else 0 end as solid_tumor
, arth as rheumatoid_arthritis
, coag as coagulopathy
, obese as obesity
, wghtloss as weight_loss
, lytes as fluid_electrolyte
, bldloss as blood_loss_anemia
, anemdef as deficiency_anemias
, alcohol as alcohol_abuse
, drug as drug_abuse
, psych as psychoses
, depress as depression

FROM `physionet-data.mimiciii_clinical.admissions` adm
left join eligrp eli
  on adm.hadm_id = eli.hadm_id
order by adm.hadm_id;       
    """
    
    job_config = bigquery.QueryJobConfig(destination=destination_table,
                                         write_disposition="WRITE_TRUNCATE")  # Overwrites existing table
    
    # Run the query
    query_job = client.query(query, job_config=job_config)
    query_job.result()  # Wait for job to complete
    
    print(f"Data successfully saved to {destination_table}")

# Run the function
save_elixhauser_to_bq()


query_demog = """
SELECT ad.subject_id, ad.hadm_id, i.icustay_id,
       UNIX_SECONDS(TIMESTAMP(ad.admittime)) AS admittime,
       UNIX_SECONDS(TIMESTAMP(ad.dischtime)) AS dischtime,
       ROW_NUMBER() OVER (PARTITION BY ad.subject_id ORDER BY i.intime ASC) AS adm_order,
       CASE 
         WHEN i.first_careunit = 'NICU' THEN 5
         WHEN i.first_careunit = 'SICU' THEN 2
         WHEN i.first_careunit = 'CSRU' THEN 4
         WHEN i.first_careunit = 'CCU' THEN 6
         WHEN i.first_careunit = 'MICU' THEN 1
         WHEN i.first_careunit = 'TSICU' THEN 3
       END AS unit,
       UNIX_SECONDS(TIMESTAMP(i.intime)) AS intime,
       UNIX_SECONDS(TIMESTAMP(i.outtime)) AS outtime,
       i.los,
       TIMESTAMP_DIFF(TIMESTAMP(i.intime), TIMESTAMP(p.dob), SECOND) / 86400 AS age,
       UNIX_SECONDS(TIMESTAMP(p.dob)) AS dob,
       UNIX_SECONDS(TIMESTAMP(p.dod)) AS dod,
       p.expire_flag,
       CASE 
         WHEN p.gender = 'M' THEN 1
         WHEN p.gender = 'F' THEN 2
       END AS gender,
       CAST(TIMESTAMP_DIFF(TIMESTAMP(p.dod), TIMESTAMP(ad.dischtime), SECOND) <= 24 * 3600 AS INT64) AS morta_hosp,
       CAST(TIMESTAMP_DIFF(TIMESTAMP(p.dod), TIMESTAMP(i.intime), SECOND) <= 90 * 24 * 3600 AS INT64) AS morta_90,
       congestive_heart_failure + cardiac_arrhythmias + valvular_disease + pulmonary_circulation + peripheral_vascular +
       hypertension + paralysis + other_neurological + chronic_pulmonary + diabetes_uncomplicated + diabetes_complicated +
       hypothyroidism + renal_failure + liver_disease + peptic_ulcer + aids + lymphoma + metastatic_cancer + solid_tumor +
       rheumatoid_arthritis + coagulopathy + obesity + weight_loss + fluid_electrolyte + blood_loss_anemia +
       deficiency_anemias + alcohol_abuse + drug_abuse + psychoses + depression AS elixhauser
FROM `mimic3_v1_4.ADMISSIONS` ad
JOIN `mimic3_v1_4.ICUSTAYS` i ON ad.hadm_id = i.hadm_id
JOIN `mimic3_v1_4.PATIENTS` p ON p.subject_id = i.subject_id
JOIN `public.ELIXHAUSER_QUAN` elix ON elix.hadm_id = ad.hadm_id
ORDER BY subject_id ASC, intime ASC
"""

extract_and_save(query_demog, 'demog.csv')

# 5. ce (Patient vitals from chartevents)
# Divided into 10 chunks for speed (indexed by ICU stay ID). Each chunk is around 170 MB.
# Each itemid here corresponds to single measurement type
for i in range(0, 100000, 10000):
    print(i)
    query = f"""
    SELECT DISTINCT icustay_id, 
           UNIX_SECONDS(TIMESTAMP(charttime)) AS charttime, 
           itemid, 
           CASE 
             WHEN value = 'None' THEN '0'
             WHEN value = 'Ventilator' THEN '1'
             WHEN value = 'Cannula' THEN '2'
             WHEN value = 'Nasal Cannula' THEN '2'
             WHEN value = 'Face Tent' THEN '3'
             WHEN value = 'Aerosol-Cool' THEN '4'
             WHEN value = 'Trach Mask' THEN '5'
             WHEN value = 'Hi Flow Neb' THEN '6'
             WHEN value = 'Non-Rebreather' THEN '7'
             WHEN value = '' THEN '8'
             WHEN value = 'Venti Mask' THEN '9'
             WHEN value = 'Medium Conc Mask' THEN '10'
             ELSE CAST(valuenum AS STRING)
           END AS valuenum
    FROM `mimic3_v1_4.CHARTEVENTS`
    WHERE icustay_id >= {200000 + i} AND icustay_id < {210000 + i}
      AND value IS NOT NULL
      AND itemid IN (467, 470, 471, 223834, 227287, 194, 224691, 226707, 226730, 581, 580, 224639, 226512, 198, 228096, 
                     211, 220045, 220179, 225309, 6701, 6, 227243, 224167, 51, 455, 220181, 220052, 225312, 224322, 6702, 
                     443, 52, 456, 8368, 8441, 225310, 8555, 8440, 220210, 3337, 224422, 618, 3603, 615, 220277, 646, 834, 
                     3655, 223762, 223761, 678, 220074, 113, 492, 491, 8448, 116, 1372, 1366, 228368, 228177, 626, 223835, 
                     3420, 160, 727, 190, 220339, 506, 505, 224700, 224686, 224684, 684, 224421, 224687, 450, 448, 445, 
                     224697, 444, 224695, 535, 224696, 543, 3083, 2566, 654, 3050, 681, 2311)
    ORDER BY icustay_id, charttime
    """
    
    extract_and_save(query, f'ce{str(i)}{str(i+10000)}.csv')

# 6. labs_ce (Labs from chartevents)
# Each itemid here corresponds to a single measurement type
query = """
SELECT icustay_id, 
       UNIX_SECONDS(TIMESTAMP(charttime)) AS charttime, 
       itemid, 
       valuenum
FROM `mimic3_v1_4.CHARTEVENTS`
WHERE valuenum IS NOT NULL 
  AND icustay_id IS NOT NULL 
  AND itemid IN (829, 1535, 227442, 227464, 4195, 3726, 3792, 837, 220645, 4194, 
                 3725, 3803, 226534, 1536, 4195, 3726, 788, 220602, 1523, 4193, 3724,
                 226536, 3747, 225664, 807, 811, 1529, 220621, 226537, 3744, 781, 1162, 225624, 
                 3737, 791, 1525, 220615, 3750, 821, 1532, 220635, 786, 225625, 1522, 3746, 816, 225667, 
                 3766, 777, 787, 770, 3801, 769, 3802, 1538, 848, 225690, 803, 1527, 225651, 3807, 
                 1539, 849, 772, 1521, 227456, 3727, 227429, 851, 227444, 814, 220228, 813, 
                 220545, 3761, 226540, 4197, 3799, 1127, 1542, 220546, 4200, 3834, 828, 227457, 
                 3789, 825, 1533, 227466, 3796, 824, 1286, 1671, 1520, 768, 220507, 815, 1530, 227467, 780, 
                 1126, 3839, 4753, 779, 490, 3785, 3838, 3837, 778, 3784, 3836, 3835, 776, 224828, 3736, 
                 4196, 3740, 74, 225668, 1531, 227443, 1817, 228640, 823, 227686)
ORDER BY icustay_id, charttime, itemid
"""

extract_and_save(query, 'labs_ce.csv')


# 7. labs_le (Labs from lab events)
query = """
SELECT xx.icustay_id, 
       UNIX_SECONDS(TIMESTAMP(f.charttime)) AS timestp, 
       f.itemid, 
       f.valuenum
FROM (
  SELECT subject_id, hadm_id, icustay_id, intime, outtime
  FROM `mimic3_v1_4.ICUSTAYS`
  GROUP BY subject_id, hadm_id, icustay_id, intime, outtime
) AS xx
INNER JOIN `mimic3_v1_4.LABEVENTS` AS f 
ON f.hadm_id = xx.hadm_id 
AND f.charttime >= TIMESTAMP_SUB(xx.intime, INTERVAL 1 DAY)
AND f.charttime <= TIMESTAMP_ADD(xx.outtime, INTERVAL 1 DAY)
AND f.itemid IN (50971, 50822, 50824, 50806, 50931, 51081, 50885, 51003, 51222,
                 50810, 51301, 50983, 50902, 50809, 51006, 50912, 50960, 50893, 50808, 50804, 50878, 50861, 51464, 50883, 50976, 50862, 51002, 50889,
                 50811, 51221, 51279, 51300, 51265, 51275, 51274, 51237, 50820, 50821, 50818, 50802, 50813, 50882, 50803)
AND f.valuenum IS NOT NULL
ORDER BY f.hadm_id, timestp, f.itemid
"""

extract_and_save(query, 'labs_le.csv')


# 8. uo (Real-time Urine Output)
query = """
SELECT icustay_id, 
       UNIX_SECONDS(TIMESTAMP(charttime)) AS charttime, 
       itemid, 
       value
FROM `mimic3_v1_4.OUTPUTEVENTS`
WHERE icustay_id IS NOT NULL 
  AND value IS NOT NULL 
  AND itemid IN (40055, 43175, 40069, 40094, 40715,
                 40473, 40085, 40057, 40056, 40405, 40428, 40096, 40651, 226559, 226560, 227510, 226561, 227489,
                 226584, 226563, 226564, 226565, 226557, 226558)
ORDER BY icustay_id, charttime, itemid
"""

extract_and_save(query, 'uo.csv')


# 9. preadm_uo (Pre-admission Urine Output)
query = """
SELECT DISTINCT oe.icustay_id, 
       UNIX_SECONDS(TIMESTAMP(oe.charttime)) AS charttime, 
       oe.itemid, 
       oe.value, 
       TIMESTAMP_DIFF(TIMESTAMP(ic.intime), TIMESTAMP(oe.charttime), MINUTE) AS datediff_minutes
FROM `mimic3_v1_4.OUTPUTEVENTS` oe
JOIN `mimic3_v1_4.ICUSTAYS` ic ON oe.icustay_id = ic.icustay_id
WHERE oe.itemid IN (40060, 226633)
ORDER BY icustay_id, charttime, itemid
"""

extract_and_save(query, 'preadm_uo.csv')


# 10. fluid_mv (Real-time input from metavision)
# This extraction converts the different rates and dimensions to a common unit
"""
Records with no rate = STAT
Records with rate = INFUSION
fluids corrected for tonicity
"""
query = """
WITH t1 AS (
  SELECT icustay_id, 
         UNIX_SECONDS(TIMESTAMP(starttime)) AS starttime, 
         UNIX_SECONDS(TIMESTAMP(endtime)) AS endtime, 
         itemid, 
         amount, 
         rate,
         CASE 
           WHEN itemid IN (30176, 30315) THEN amount * 0.25
           WHEN itemid IN (30161) THEN amount * 0.3
           WHEN itemid IN (30020, 30015, 225823, 30321, 30186, 30211, 30353, 42742, 42244, 225159) THEN amount * 0.5
           WHEN itemid IN (227531) THEN amount * 2.75
           WHEN itemid IN (30143, 225161) THEN amount * 3
           WHEN itemid IN (30009, 220862) THEN amount * 5
           WHEN itemid IN (30030, 220995, 227533) THEN amount * 6.66
           WHEN itemid IN (228341) THEN amount * 8
           ELSE amount 
         END AS tev -- total equivalent volume
  FROM `mimic3_v1_4.INPUTEVENTS_MV`
  WHERE icustay_id IS NOT NULL 
    AND amount IS NOT NULL 
    AND itemid IN (225158, 225943, 226089, 225168, 225828, 220862, 220970, 220864, 225159, 220995, 225170, 225825, 227533, 225161, 227531, 225171, 225827, 225941, 225823, 228341, 30018, 30021, 30015, 30296, 30020, 30066, 30001, 30030, 30060, 30005, 30321, 30006, 30061, 30009, 30179, 30190, 30143, 30160, 30008, 30168, 30186, 30211, 30353, 30159, 30007, 30185, 30063, 30094, 30352, 30014, 30011, 30210, 46493, 45399, 46516, 40850, 30176, 30161, 30381, 30315, 42742, 30180, 46087, 41491, 30004, 42698, 42244)
)
SELECT icustay_id, 
       starttime, 
       endtime, 
       itemid, 
       ROUND(CAST(amount AS NUMERIC), 3) AS amount,
       ROUND(CAST(rate AS NUMERIC), 3) AS rate,
       ROUND(CAST(tev AS NUMERIC), 3) AS tev -- total equivalent volume
FROM t1
ORDER BY icustay_id, starttime, itemid
"""

extract_and_save(query, 'fluid_mv.csv')

# 11. fluid_cv (Real-time input from carevue)
# This extraction converts the different rates and dimensions to a common units
"""
In CAREVUE, all records are considered STAT doses!!
fluids corrected for tonicity
"""
query = """
WITH t1 AS (
  SELECT icustay_id, 
         UNIX_SECONDS(TIMESTAMP(charttime)) AS charttime, 
         itemid, 
         amount,
         CASE 
           WHEN itemid IN (30176, 30315) THEN amount * 0.25
           WHEN itemid IN (30161) THEN amount * 0.3
           WHEN itemid IN (30020, 30321, 30015, 225823, 30186, 30211, 30353, 42742, 42244, 225159) THEN amount * 0.5
           WHEN itemid IN (227531) THEN amount * 2.75
           WHEN itemid IN (30143, 225161) THEN amount * 3
           WHEN itemid IN (30009, 220862) THEN amount * 5
           WHEN itemid IN (30030, 220995, 227533) THEN amount * 6.66
           WHEN itemid IN (228341) THEN amount * 8
           ELSE amount 
         END AS tev -- total equivalent volume
  FROM `mimic3_v1_4.INPUTEVENTS_CV`
  WHERE amount IS NOT NULL 
    AND itemid IN (225158, 225943, 226089, 225168, 225828, 220862, 220970, 220864, 225159, 220995, 225170, 227533, 225161, 227531, 225171, 225827, 225941, 225823, 225825, 228341, 30018, 30021, 30015, 30296, 30020, 30066, 30001, 30030, 30060, 30005, 30321, 30006, 30061, 30009, 30179, 30190, 30143, 30160, 30008, 30168, 30186, 30211, 30353, 30159, 30007, 30185, 30063, 30094, 30352, 30014, 30011, 30210, 46493, 45399, 46516, 40850, 30176, 30161, 30381, 30315, 42742, 30180, 46087, 41491, 30004, 42698, 42244)
)
SELECT icustay_id, 
       charttime, 
       itemid, 
       ROUND(CAST(amount AS NUMERIC), 3) AS amount, 
       ROUND(CAST(tev AS NUMERIC), 3) AS tev -- total equivalent volume
FROM t1
ORDER BY icustay_id, charttime, itemid
"""

extract_and_save(query, 'fluid_cv.csv')

# 12. preadm_fluid (Pre-admission fluid intake)
query = """
WITH mv AS (
  SELECT ie.icustay_id, SUM(ie.amount) AS sum
  FROM `mimic3_v1_4.INPUTEVENTS_MV` ie
  WHERE ie.itemid IN (30054, 30055, 30101, 30102, 30103, 30104, 30105, 30108, 226361,
                      226363, 226364, 226365, 226367, 226368, 226369, 226370, 226371, 226372, 226375, 226376, 227070, 227071, 227072)
  GROUP BY icustay_id
), cv AS (
  SELECT ie.icustay_id, SUM(ie.amount) AS sum
  FROM `mimic3_v1_4.INPUTEVENTS_CV` ie
  WHERE ie.itemid IN (30054, 30055, 30101, 30102, 30103, 30104, 30105, 30108, 226361,
                      226363, 226364, 226365, 226367, 226368, 226369, 226370, 226371, 226372, 226375, 226376, 227070, 227071, 227072)
  GROUP BY icustay_id
)

SELECT pt.icustay_id,
       CASE 
         WHEN mv.sum IS NOT NULL THEN mv.sum
         WHEN cv.sum IS NOT NULL THEN cv.sum
         ELSE NULL 
       END AS inputpreadm
FROM `mimic3_v1_4.ICUSTAYS` pt
LEFT JOIN mv ON mv.icustay_id = pt.icustay_id
LEFT JOIN cv ON cv.icustay_id = pt.icustay_id
ORDER BY icustay_id
"""

extract_and_save(query, 'preadm_fluid.csv')


# 13. vaso_mv (Vasopressors from metavision)
# This extraction converts the different rates and dimensions to a common units
"""
Drugs converted in noradrenaline-equivalent
Body weight assumed 80 kg when missing
"""
query = """
SELECT icustay_id, 
       itemid, 
       UNIX_SECONDS(TIMESTAMP(starttime)) AS starttime, 
       UNIX_SECONDS(TIMESTAMP(endtime)) AS endtime,
       CASE 
         WHEN itemid IN (30120, 221906, 30047) AND rateuom = 'mcg/kg/min' THEN ROUND(CAST(rate AS NUMERIC), 3)  -- norad
         WHEN itemid IN (30120, 221906, 30047) AND rateuom = 'mcg/min' THEN ROUND(CAST(rate / 80 AS NUMERIC), 3)  -- norad
         WHEN itemid IN (30119, 221289) AND rateuom = 'mcg/kg/min' THEN ROUND(CAST(rate AS NUMERIC), 3) -- epi
         WHEN itemid IN (30119, 221289) AND rateuom = 'mcg/min' THEN ROUND(CAST(rate / 80 AS NUMERIC), 3) -- epi
         WHEN itemid IN (30051, 222315) AND rate > 0.2 THEN ROUND(CAST(rate * 5 / 60 AS NUMERIC), 3) -- vasopressin, in U/h
         WHEN itemid IN (30051, 222315) AND rateuom = 'units/min' THEN ROUND(CAST(rate * 5 AS NUMERIC), 3) -- vasopressin
         WHEN itemid IN (30051, 222315) AND rateuom = 'units/hour' THEN ROUND(CAST(rate * 5 / 60 AS NUMERIC), 3) -- vasopressin
         WHEN itemid IN (30128, 221749, 30127) AND rateuom = 'mcg/kg/min' THEN ROUND(CAST(rate * 0.45 AS NUMERIC), 3) -- phenyl
         WHEN itemid IN (30128, 221749, 30127) AND rateuom = 'mcg/min' THEN ROUND(CAST(rate * 0.45 / 80 AS NUMERIC), 3) -- phenyl
         WHEN itemid IN (221662, 30043, 30307) AND rateuom = 'mcg/kg/min' THEN ROUND(CAST(rate * 0.01 AS NUMERIC), 3)  -- dopa
         WHEN itemid IN (221662, 30043, 30307) AND rateuom = 'mcg/min' THEN ROUND(CAST(rate * 0.01 / 80 AS NUMERIC), 3) 
         ELSE NULL 
       END AS rate_std -- dopa
FROM `mimic3_v1_4.INPUTEVENTS_MV`
WHERE itemid IN (30128, 30120, 30051, 221749, 221906, 30119, 30047, 
                 30127, 221289, 222315, 221662, 30043, 30307) 
  AND rate IS NOT NULL 
  AND statusdescription <> 'Rewritten'
ORDER BY icustay_id, itemid, starttime
"""

extract_and_save(query, 'vaso_mv.csv')


# 14. vaso_cv (Vasopressors from carevue)
# This extraction converts the different rates and dimensions to a common units
"""
Same comments as above
"""
query = """
SELECT icustay_id, 
       itemid, 
       UNIX_SECONDS(TIMESTAMP(charttime)) AS charttime,
       CASE 
         WHEN itemid IN (30120, 221906, 30047) AND rateuom = 'mcgkgmin' THEN ROUND(CAST(rate AS NUMERIC), 3) -- norad
         WHEN itemid IN (30120, 221906, 30047) AND rateuom = 'mcgmin' THEN ROUND(CAST(rate / 80 AS NUMERIC), 3)  -- norad
         WHEN itemid IN (30119, 221289) AND rateuom = 'mcgkgmin' THEN ROUND(CAST(rate AS NUMERIC), 3) -- epi
         WHEN itemid IN (30119, 221289) AND rateuom = 'mcgmin' THEN ROUND(CAST(rate / 80 AS NUMERIC), 3) -- epi
         WHEN itemid IN (30051, 222315) AND rate > 0.2 THEN ROUND(CAST(rate * 5 / 60 AS NUMERIC), 3) -- vasopressin, in U/h
         WHEN itemid IN (30051, 222315) AND rateuom = 'Umin' AND rate < 0.2 THEN ROUND(CAST(rate * 5 AS NUMERIC), 3) -- vasopressin
         WHEN itemid IN (30051, 222315) AND rateuom = 'Uhr' THEN ROUND(CAST(rate * 5 / 60 AS NUMERIC), 3) -- vasopressin
         WHEN itemid IN (30128, 221749, 30127) AND rateuom = 'mcgkgmin' THEN ROUND(CAST(rate * 0.45 AS NUMERIC), 3) -- phenyl
         WHEN itemid IN (30128, 221749, 30127) AND rateuom = 'mcgmin' THEN ROUND(CAST(rate * 0.45 / 80 AS NUMERIC), 3) -- phenyl
         WHEN itemid IN (221662, 30043, 30307) AND rateuom = 'mcgkgmin' THEN ROUND(CAST(rate * 0.01 AS NUMERIC), 3) -- dopa
         WHEN itemid IN (221662, 30043, 30307) AND rateuom = 'mcgmin' THEN ROUND(CAST(rate * 0.01 / 80 AS NUMERIC), 3) 
         ELSE NULL 
       END AS rate_std -- dopa
FROM `mimic3_v1_4.INPUTEVENTS_CV`
WHERE itemid IN (30128, 30120, 30051, 221749, 221906, 30119, 30047, 30127, 221289, 222315, 221662, 30043, 30307) 
  AND rate IS NOT NULL
ORDER BY icustay_id, itemid, charttime
"""

extract_and_save(query, 'vaso_cv.csv')


# 15. mechvent (Mechanical ventilation)
query = """
SELECT
    icustay_id, 
    UNIX_SECONDS(TIMESTAMP(charttime)) AS charttime,
    MAX(
      CASE
        WHEN itemid IS NULL OR value IS NULL THEN 0 -- can't have null values
        WHEN itemid = 720 AND value != 'Other/Remarks' THEN 1  -- VentTypeRecorded
        WHEN itemid = 467 AND value = 'Ventilator' THEN 1 -- O2 delivery device == ventilator
        WHEN itemid IN (
          445, 448, 449, 450, 1340, 1486, 1600, 224687, -- minute volume
          639, 654, 681, 682, 683, 684, 224685, 224684, 224686, -- tidal volume
          218, 436, 535, 444, 459, 224697, 224695, 224696, 224746, 224747, -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
          221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187, -- Insp pressure
          543, -- PlateauPressure
          5865, 5866, 224707, 224709, 224705, 224706, -- APRV pressure
          60, 437, 505, 506, 686, 220339, 224700, -- PEEP
          3459, -- high pressure relief
          501, 502, 503, 224702, -- PCV
          223, 667, 668, 669, 670, 671, 672, -- TCPCV
          157, 158, 1852, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 8382, 227809, 227810, -- ETT
          224701 -- PSVlevel
        ) THEN 1
        ELSE 0
      END
    ) AS mechvent,
    MAX(
      CASE 
        WHEN itemid IS NULL OR value IS NULL THEN 0
        WHEN itemid = 640 AND value = 'Extubated' THEN 1
        WHEN itemid = 640 AND value = 'Self Extubation' THEN 1
        ELSE 0
      END
    ) AS extubated,
    MAX(
      CASE 
        WHEN itemid IS NULL OR value IS NULL THEN 0
        WHEN itemid = 640 AND value = 'Self Extubation' THEN 1
        ELSE 0
      END
    ) AS selfextubated
FROM `mimic3_v1_4.CHARTEVENTS` ce
WHERE value IS NOT NULL
  AND itemid IN (
      640, -- extubated
      720, -- vent type
      467, -- O2 delivery device
      445, 448, 449, 450, 1340, 1486, 1600, 224687, -- minute volume
      639, 654, 681, 682, 683, 684, 224685, 224684, 224686, -- tidal volume
      218, 436, 535, 444, 459, 224697, 224695, 224696, 224746, 224747, -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
      221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187, -- Insp pressure
      543, -- PlateauPressure
      5865, 5866, 224707, 224709, 224705, 224706, -- APRV pressure
      60, 437, 505, 506, 686, 220339, 224700, -- PEEP
      3459, -- high pressure relief
      501, 502, 503, 224702, -- PCV
      223, 667, 668, 669, 670, 671, 672, -- TCPCV
      157, 158, 1852, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 8382, 227809, 227810, -- ETT
      224701 -- PSVlevel
  )
GROUP BY icustay_id, charttime
"""

extract_and_save(query, 'mechvent.csv')
