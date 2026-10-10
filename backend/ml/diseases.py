"""Disease registry. Feature schemas are NOT hard-coded: they are derived from each real dataset
at training time, so diseases never share (or mix) a schema."""

DISEASES = {
    "cvd": dict(name="Cardiovascular Disease", short="CVD", icon="heart",
                target_aliases=["target", "heartdisease", "heart_disease", "cardio", "output", "num",
                               "cardiovascular_disease", "tenyearchd", "has_heart_disease"],
                fn_cost="A missed case may delay care for a life-threatening condition.",
                fp_cost="A false alarm leads to extra tests and anxiety."),
    "diabetes": dict(name="Diabetes Mellitus", short="DM", icon="drop",
                     target_aliases=["outcome", "diabetes", "diabetes_binary", "class", "diabetic", "target"],
                     fn_cost="A missed case delays glucose management and raises long-term complication risk.",
                     fp_cost="A false alarm leads to confirmatory testing (e.g. HbA1c)."),
    "breast_cancer": dict(name="Breast Cancer", short="BC", icon="ribbon",
                          target_aliases=["diagnosis", "class", "target", "malignant"],
                          fn_cost="A missed malignancy delays treatment - the most serious error.",
                          fp_cost="A false alarm leads to additional imaging/biopsy."),
    "ckd": dict(name="Chronic Kidney Disease", short="CKD", icon="kidney",
                target_aliases=["classification", "class", "ckd", "target"],
                fn_cost="A missed case delays nephrology referral and progression management.",
                fp_cost="A false alarm leads to repeat laboratory testing."),
    "tb": dict(name="Tuberculosis", short="TB", icon="lungs",
               target_aliases=["tb", "tuberculosis", "target", "label", "class", "outcome", "status"],
               fn_cost="A missed case allows ongoing transmission and disease progression.",
               fp_cost="A false alarm leads to confirmatory sputum/molecular tests."),
}

POS_TOKENS = {"1", "1.0", "yes", "y", "true", "t", "ckd", "m", "malignant", "positive", "pos", "tb",
              "tuberculosis", "present", "abnormal", "diabetic", "disease", "sick"}
NEG_TOKENS = {"0", "0.0", "no", "n", "false", "f", "notckd", "b", "benign", "negative", "neg", "normal",
              "absent", "healthy", "non_diabetic", "nondiabetic", "no_disease", "not_tb", "notb"}

# Domain plausibility rules: (regex on normalised column, low, high). Applied only when the column's
# median lies inside [low, high] (i.e. it is on that unit scale; e.g. cardio's 1-3 cholesterol is skipped).
DOMAIN_RULES = [
    (r"^ap_hi$", 60, 250), (r"^ap_lo$", 30, 160), (r"^height$", 100, 230), (r"^weight$", 25, 250),
    (r"^age$", 0, 120), (r"^bmi$", 10, 80),
    (r"^(glucose|plasma_glucose|bgr|blood_glucose_random|fasting_blood_sugar_mgdl)$", 20, 800),
    (r"(blood_pressure|^bp$|bloodpressure|trestbps|ap_hi|ap_lo|diastolic|systolic)", 20, 300),
    (r"^(chol|cholesterol|serum_cholesterol)$", 50, 700),
    (r"^(skin_?thickness)$", 1, 100), (r"^insulin$", 1, 1000),
]
