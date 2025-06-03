#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv('NYPD_Complaint_Data_Historic_20250515_preprocessed.csv', low_memory=False)


# In[2]:


df['LAW_CAT_CD']


# In[48]:


html_snippet="""
<table style="border-collapse:collapse; width:100%; font-family:Arial, Helvetica, sans-serif; font-size:14px; color:#999;">
  <thead>
    <tr style="background:#f2f2f2;">
      <th style="border:1px solid #ddd; padding:8px; text-align:left;">Column&nbsp;Name</th>
      <th style="border:1px solid #ddd; padding:8px; text-align:left;">API&nbsp;Field&nbsp;Name</th>
      <th style="border:1px solid #ddd; padding:8px; text-align:left;">Description</th>
      <th style="border:1px solid #ddd; padding:8px; text-align:left;">Data&nbsp;Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">CMPLNT_NUM</td>
      <td style="border:1px solid #ddd; padding:6px;">cmplnt_num</td>
      <td style="border:1px solid #ddd; padding:6px;">Randomly&nbsp;generated persistent ID for each complaint</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">CMPLNT_FR_DT</td>
      <td style="border:1px solid #ddd; padding:6px;">cmplnt_fr_dt</td>
      <td style="border:1px solid #ddd; padding:6px;">Date of occurrence (start date)</td>
      <td style="border:1px solid #ddd; padding:6px;">Floating&nbsp;Timestamp</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">CMPLNT_FR_TM</td>
      <td style="border:1px solid #ddd; padding:6px;">cmplnt_fr_tm</td>
      <td style="border:1px solid #ddd; padding:6px;">Time of occurrence (start time)</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">CMPLNT_TO_DT</td>
      <td style="border:1px solid #ddd; padding:6px;">cmplnt_to_dt</td>
      <td style="border:1px solid #ddd; padding:6px;">Date of occurrence (end date, if applicable)</td>
      <td style="border:1px solid #ddd; padding:6px;">Floating&nbsp;Timestamp</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">CMPLNT_TO_TM</td>
      <td style="border:1px solid #ddd; padding:6px;">cmplnt_to_tm</td>
      <td style="border:1px solid #ddd; padding:6px;">Time of occurrence (end time, if applicable)</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">ADDR_PCT_CD</td>
      <td style="border:1px solid #ddd; padding:6px;">addr_pct_cd</td>
      <td style="border:1px solid #ddd; padding:6px;">Precinct where incident occurred</td>
      <td style="border:1px solid #ddd; padding:6px;">Number</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">RPT_DT</td>
      <td style="border:1px solid #ddd; padding:6px;">rpt_dt</td>
      <td style="border:1px solid #ddd; padding:6px;">Date event was reported to police</td>
      <td style="border:1px solid #ddd; padding:6px;">Floating&nbsp;Timestamp</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">KY_CD</td>
      <td style="border:1px solid #ddd; padding:6px;">ky_cd</td>
      <td style="border:1px solid #ddd; padding:6px;">Three-digit offense classification code</td>
      <td style="border:1px solid #ddd; padding:6px;">Number</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">OFNS_DESC</td>
      <td style="border:1px solid #ddd; padding:6px;">ofns_desc</td>
      <td style="border:1px solid #ddd; padding:6px;">Offense description (by key code)</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">PD_CD</td>
      <td style="border:1px solid #ddd; padding:6px;">pd_cd</td>
      <td style="border:1px solid #ddd; padding:6px;">Three-digit internal classification code</td>
      <td style="border:1px solid #ddd; padding:6px;">Number</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">PD_DESC</td>
      <td style="border:1px solid #ddd; padding:6px;">pd_desc</td>
      <td style="border:1px solid #ddd; padding:6px;">Internal classification description</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">CRM_ATPT_CPTD_CD</td>
      <td style="border:1px solid #ddd; padding:6px;">crm_atpt_cptd_cd</td>
      <td style="border:1px solid #ddd; padding:6px;">Attempted vs. completed indicator</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">LAW_CAT_CD</td>
      <td style="border:1px solid #ddd; padding:6px;">law_cat_cd</td>
      <td style="border:1px solid #ddd; padding:6px;">Offense level (felony, misdemeanor, violation)</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">BORO_NM</td>
      <td style="border:1px solid #ddd; padding:6px;">boro_nm</td>
      <td style="border:1px solid #ddd; padding:6px;">Borough name</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">LOC_OF_OCCUR_DESC</td>
      <td style="border:1px solid #ddd; padding:6px;">loc_of_occur_desc</td>
      <td style="border:1px solid #ddd; padding:6px;">Specific location (inside, front of, etc.)</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">PREM_TYP_DESC</td>
      <td style="border:1px solid #ddd; padding:6px;">prem_typ_desc</td>
      <td style="border:1px solid #ddd; padding:6px;">Premises description (street, residence, etc.)</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">JURIS_DESC</td>
      <td style="border:1px solid #ddd; padding:6px;">juris_desc</td>
      <td style="border:1px solid #ddd; padding:6px;">Jurisdiction description</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">JURISDICTION_CODE</td>
      <td style="border:1px solid #ddd; padding:6px;">jurisdiction_code</td>
      <td style="border:1px solid #ddd; padding:6px;">Jurisdiction responsible (0 Police, 1 Transit, etc.)</td>
      <td style="border:1px solid #ddd; padding:6px;">Number</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">PARKS_NM</td>
      <td style="border:1px solid #ddd; padding:6px;">parks_nm</td>
      <td style="border:1px solid #ddd; padding:6px;">NYC park/playground name (if applicable)</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">HADEVELOPT</td>
      <td style="border:1px solid #ddd; padding:6px;">hadevelopt</td>
      <td style="border:1px solid #ddd; padding:6px;">NYCHA housing development name (if applicable)</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">HOUSING_PSA</td>
      <td style="border:1px solid #ddd; padding:6px;">housing_psa</td>
      <td style="border:1px solid #ddd; padding:6px;">Development-level code</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">X_COORD_CD</td>
      <td style="border:1px solid #ddd; padding:6px;">x_coord_cd</td>
      <td style="border:1px solid #ddd; padding:6px;">X-coordinate (NY State Plane, feet)</td>
      <td style="border:1px solid #ddd; padding:6px;">Number</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">Y_COORD_CD</td>
      <td style="border:1px solid #ddd; padding:6px;">y_coord_cd</td>
      <td style="border:1px solid #ddd; padding:6px;">Y-coordinate (NY State Plane, feet)</td>
      <td style="border:1px solid #ddd; padding:6px;">Number</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">SUSP_AGE_GROUP</td>
      <td style="border:1px solid #ddd; padding:6px;">susp_age_group</td>
      <td style="border:1px solid #ddd; padding:6px;">Suspect age group</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">SUSP_RACE</td>
      <td style="border:1px solid #ddd; padding:6px;">susp_race</td>
      <td style="border:1px solid #ddd; padding:6px;">Suspect race description</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">SUSP_SEX</td>
      <td style="border:1px solid #ddd; padding:6px;">susp_sex</td>
      <td style="border:1px solid #ddd; padding:6px;">Suspect sex description</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">TRANSIT_DISTRICT</td>
      <td style="border:1px solid #ddd; padding:6px;">transit_district</td>
      <td style="border:1px solid #ddd; padding:6px;">Transit district</td>
      <td style="border:1px solid #ddd; padding:6px;">Number</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">Latitude</td>
      <td style="border:1px solid #ddd; padding:6px;">latitude</td>
      <td style="border:1px solid #ddd; padding:6px;">Mid-block latitude (WGS 84)</td>
      <td style="border:1px solid #ddd; padding:6px;">Number</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">Longitude</td>
      <td style="border:1px solid #ddd; padding:6px;">longitude</td>
      <td style="border:1px solid #ddd; padding:6px;">Mid-block longitude (WGS 84)</td>
      <td style="border:1px solid #ddd; padding:6px;">Number</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">Lat_Lon</td>
      <td style="border:1px solid #ddd; padding:6px;">lat_lon</td>
      <td style="border:1px solid #ddd; padding:6px;">Combined latitude/longitude point</td>
      <td style="border:1px solid #ddd; padding:6px;">Location</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">PATROL_BORO</td>
      <td style="border:1px solid #ddd; padding:6px;">patrol_boro</td>
      <td style="border:1px solid #ddd; padding:6px;">Patrol borough name</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">STATION_NAME</td>
      <td style="border:1px solid #ddd; padding:6px;">station_name</td>
      <td style="border:1px solid #ddd; padding:6px;">Transit station name</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">VIC_AGE_GROUP</td>
      <td style="border:1px solid #ddd; padding:6px;">vic_age_group</td>
      <td style="border:1px solid #ddd; padding:6px;">Victim age group</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr style="background:#fafafa;">
      <td style="border:1px solid #ddd; padding:6px;">VIC_RACE</td>
      <td style="border:1px solid #ddd; padding:6px;">vic_race</td>
      <td style="border:1px solid #ddd; padding:6px;">Victim race description</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:6px;">VIC_SEX</td>
      <td style="border:1px solid #ddd; padding:6px;">vic_sex</td>
      <td style="border:1px solid #ddd; padding:6px;">Victim sex description</td>
      <td style="border:1px solid #ddd; padding:6px;">Text</td>
    </tr>
  </tbody>
</table>


 """


# In[49]:


from IPython.display import HTML, display
display(HTML(html_snippet))


# In[5]:


df.columns


# In[6]:


import requests
import json
promt = """The dataset has these columns (with context):

• CMPLNT_NUM: a unique, persistent ID for each reported complaint.  
• CMPLNT_FR_DT: the exact start date when the incident occurred.  
• CMPLNT_FR_TM: the exact start time of the incident.  
• CMPLNT_TO_DT: the end date of the incident, if it spanned multiple days or the exact time is unknown.  
• CMPLNT_TO_TM: the end time of the incident, if the exact start time is unknown.  
• ADDR_PCT_CD: the precinct number where the incident took place.  
• RPT_DT: the date the incident was reported to police.  
• KY_CD: a 3-digit code classifying the type of offense.  
• OFNS_DESC: a textual description of the offense (e.g. “ROBBERY”, “BURGLARY”).  
• PD_CD: a more granular 3-digit internal offense code.  
• PD_DESC: a textual description corresponding to PD_CD.  
• CRM_ATPT_CPTD_CD: indicator whether the crime was completed (“COMPLETED”) or only attempted (“ATTEMPTED”).  
• LAW_CAT_CD: the severity level of offense (e.g. “FELONY”, “MISDEMEANOR”, “VIOLATION”).  
• BORO_NM: the NYC borough name where the incident occurred.  
• LOC_OF_OCCUR_DESC: the specific location on or around a premises (e.g. “INSIDE”, “FRONT OF”).  
• PREM_TYP_DESC: the type of premises (e.g. “GROCERY STORE”, “STREET”, “RESIDENCE”).  
• JURIS_DESC: textual description of the jurisdiction code.  
• JURISDICTION_CODE: numeric code for the jurisdiction (0=Police, 1=Transit, 2=Housing, 3=External).  
• PARKS_NM: name of the NYC park or playground, if applicable.  
• HADEVELOPT: name of NYCHA housing development, if applicable.  
• HOUSING_PSA: development-level policing area code.  
• X_COORD_CD: X coordinate in NY State Plane (feet).  
• Y_COORD_CD: Y coordinate in NY State Plane (feet).  
• SUSP_AGE_GROUP: suspect’s age group (e.g. “25–44”).  
• SUSP_RACE: suspect’s race category (e.g. “WHITE”, “BLACK”).  
• SUSP_SEX: suspect’s sex (e.g. “M”, “F”).  
• TRANSIT_DISTRICT: numeric code of the transit district, if on transit property.  
• Latitude: geographic latitude of the incident (decimal degrees).  
• Longitude: geographic longitude of the incident (decimal degrees).  
• Lat_Lon: combined geospatial point (latitude, longitude).  
• PATROL_BORO: patrol borough name responsible for the area.  
• STATION_NAME: transit station name, if incident occurred on transit property.  
• VIC_AGE_GROUP: victim’s age group.  
• VIC_RACE: victim’s race category.  
• VIC_SEX: victim’s sex.

Please identify which of these columns should be used in code to estimate bias (i.e., protected attributes, demographic groupings, or contextual factors relevant for fairness analysis).  
Respond with only the column names, separated by commas."""


# 1. LLM point URL
url = "https://xoxof3kdzvlwkyk5hfaajacb.agents.do-ai.run/api/v1/chat/completions"

# 2. Prepare the headers with Content-Type and Authorization (Bearer token)
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer _V__VxUKW6o9wnCPGh8YYgof_Rknl-XQ"
} 
content= promt

# 3. Construct the JSON payload with the user message and options flags
payload = {
    "messages": [
        {"role": "user", "content": content}
    ],
    "stream": False,
    "include_functions_info": False,
    "include_retrieval_info": False,
    "include_guardrails_info": False
}

# 4. Send the POST request to the API
response = requests.post(url, headers=headers, json=payload)

# 5. Print the response content to the console
content =json.loads(response.text)['choices'][0]['message']['content']
print(content)

# 3. Build second prompt, injecting the first answer as context
second_prompt = f"""I have identified the following columns as potential protected/bias‐related features:

{content}

Now, among **all** the original dataset columns,provide only one columns should be treated as the **target variable** when computing precision and recall for bias evaluation?  
Respond with only the column name."""
 
# 4. Send second request—this time including the prior answer as context
payload2 = {
    "messages": [
        {"role": "user", "content": promt},
        {"role": "assistant", "content": content},
        {"role": "user", "content": second_prompt}
    ],
    "stream": False
}
resp2 = requests.post(url, headers=headers, json=payload2)

target_column = json.loads(resp2.text)['choices'][0]['message']['content']
print("Target variable:", target_column)



# In[ ]:


# Revised calculation: correlation per original feature (numeric and categorical)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Assume df, content (protected columns), and target_column are defined

# 1. Define lists
protected_columns = [c.strip() for c in content.split(',')]
exclude_cols = {'CMPLNT_NUM', 'CMPLNT_FR_DT', 'CMPLNT_FR_TM',
                'CMPLNT_TO_DT', 'CMPLNT_TO_TM',
                'Lat_Lon', 'Latitude', 'Longitude'}

# 2. Feature columns = all minus protected, target, and excluded
feature_columns = [
    col for col in df.columns
    if col not in protected_columns
    and col != target_column
    and col not in exclude_cols
]

# 3. Drop rows missing target
sub = df[feature_columns + [target_column]].dropna(subset=[target_column])[:10000]

# 4. Encode target if categorical
y = sub[target_column]
if y.dtype == 'object' or y.dtype.name == 'category':
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
else:
    y_enc = y.values

# 5. Determine numeric & categorical features
numeric_feats = sub[feature_columns].select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_feats = sub[feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()

# 6. Compute per-feature correlation with target
corr_dict = {}

# Numeric: Pearson
for col in numeric_feats:
    corr_val = abs(sub[col].corr(pd.Series(y_enc, index=sub.index)))
    corr_dict[col] = corr_val

# Categorical: max correlation among its dummies
for col in categorical_feats:
    dummies = pd.get_dummies(sub[col], prefix=col)
    max_corr = max(abs(dummies[c].corr(pd.Series(y_enc, index=sub.index))) for c in dummies.columns)
    corr_dict[col] = max_corr

# 7. Create sorted series and take top 10
corr_series = pd.Series(corr_dict).sort_values(ascending=False)
top10 = corr_series.head(10)

# 8. Format snippet
corr_snippet = "\n".join(f"{var}={val:.3f}" for var, val in top10.items())

# Display the snippet
print("Top 10 feature correlations with target:\n")
print(corr_snippet)

# 7. Craft the prompt
third_prompt = f"""
I have these protected/bias-related features (do NOT include them in the correlation step):
{', '.join(protected_columns)}

My target variable is:
{target_column}

Here are the top 10 features most correlated with the target, formatted as variable=correlation:
{corr_snippet}

Please respond **only** with the column names that should be excluded due to high dependence on the target—separated by commas, no extra text.
""".strip()

# 8. Call the LLM
payload3 = {
    "messages": [
        {"role": "system", "content": "You are a concise assistant that only lists column names."},
        {"role": "user",   "content": third_prompt}
    ]
}
resp3 = requests.post(url, headers=headers, json=payload3)
excluded_columns = resp3.json()['choices'][0]['message']['content']
print("Columns to exclude:", excluded_columns)


# In[30]:


# Start with your comma-separated string (no spaces), split into a list
excluded_list = excluded_columns.replace(' ', '').split(',')

# Define the extra columns you want to add
extra_cols = [
    'CMPLNT_NUM',
    'CMPLNT_FR_DT',
    'CMPLNT_FR_TM',
    'CMPLNT_TO_DT',
    'CMPLNT_TO_TM',
    'Lat_Lon',
    'Latitude',
    'Longitude'
]

# Extend the original list with your extras
excluded_list.extend(extra_cols)


# In[32]:


excluded_list


# In[49]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def analyze_biases(df: pd.DataFrame, content: str):
    """
    Analyze potential biases in a dataset based on the specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the relevant columns.
    content : str
        Comma-separated column names to include in the analysis, e.g.
        "SUSP_RACE, SUSP_SEX, VIC_RACE, VIC_SEX, SUSP_AGE_GROUP, VIC_AGE_GROUP, BORO_NM, PATROL_BORO"
    """
    # Parse the column names
    cols = [col.strip() for col in content.split(',')]

    # 1. Univariate distributions
    for col in cols:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found in DataFrame.")
            continue
        dist = df[col].value_counts(normalize=True, dropna=False).sort_values(ascending=False)
        print(f"\n— Distribution for {col} —")
        print(dist.to_frame('proportion'))

        plt.figure()
        dist.plot(kind='bar')
        plt.title(f'Normalized Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Proportion')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    

analyze_biases(df, content)



# In[41]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def evaluate_model_bias(df: pd.DataFrame,
                        target_col: str,
                        protected_attr: str,
                        features: list = None,
                        test_size: float = 0.3,
                        random_state: int = 42):
    """
    Trains a logistic regression (binary or multinomial) to predict a categorical target 
    and computes precision & recall per protected group for each class.
    Handles missing values by imputing numeric and categorical features and ensures
    consistent class labels for group-level reports.
    """
    # Drop missing in target and protected_attr
    data = df.dropna(subset=[target_col, protected_attr])
    
    # Auto-select features
    if features is None:
        exclude = {target_col, protected_attr,
                   'CMPLNT_NUM', 'CMPLNT_FR_DT', 'CMPLNT_FR_TM',
                   'CMPLNT_TO_DT', 'CMPLNT_TO_TM',
                   'Lat_Lon', 'Latitude', 'Longitude'}
        features = [c for c in df.columns if c not in exclude]
    
    X = data[features]
    y = data[target_col].astype(str)
    groups = data[protected_attr].astype(str)
    
    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(range(len(le.classes_)))
    class_names = le.classes_
    
    # Pipelines
    num_feats = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_feats = X.select_dtypes(include=['object','category']).columns.tolist()
    numeric_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_feats),
        ('cat', categorical_pipeline, cat_feats)
    ])
    
    X_proc = preprocessor.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X_proc, y_enc, groups, test_size=test_size,
        stratify=y_enc, random_state=random_state
    )
    
    # Train
    model = LogisticRegression(max_iter=1000, multi_class='auto')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Overall report
    overall = pd.DataFrame(
        classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    ).transpose()[['precision','recall','f1-score']]
    print("=== Overall Classification Report ===")
    print(overall)
    
    # Group-wise report
    rows = []
    for grp in sorted(g_test.unique()):
        mask = g_test == grp
        if mask.sum() == 0:
            continue
        grp_dict = classification_report(
            y_test[mask], y_pred[mask],
            labels=classes, target_names=class_names, output_dict=True
        )
        for cls_name in class_names:
            cls_metrics = grp_dict.get(cls_name, {})
            rows.append({
                protected_attr: grp,
                'class': cls_name,
                'precision': cls_metrics.get('precision', 0.0),
                'recall': cls_metrics.get('recall', 0.0),
                'support': cls_metrics.get('support', 0)
            })
    group_report = pd.DataFrame(rows)
    print(f"\n=== Metrics by {protected_attr} and Class ===")
    display(group_report)
    
    return model, preprocessor, overall, group_report


# In[42]:


# Build a mask: True for cols *not* in excluded_list
mask = ~df.columns.isin(excluded_list)

# Apply it to select only the desired columns
df_clean = df.loc[:, mask]


# In[43]:


for i in content.split(','):
    i=i.replace(' ', '')
    print (i)
    model, preproc, overall, by_race = evaluate_model_bias(
        df_clean[:10000],
        target_col=target_column,   
        protected_attr=i
    )


# In[ ]:




