README - Email Classification System

This project implements multiple machine learning models and a hybrid BERT + XGBoost model for email classification into:
- Spam
- Ham
- Promotional
- Malware

----------------------------------------
SETUP INSTRUCTIONS
----------------------------------------

1. (Optional but Recommended) Create Virtual Environment:

python -m venv venv
venv\Scripts\activate

----------------------------------------
2. Install Required Packages (Working Versions)

numpy==1.26.4
pandas==2.3.3
scikit-learn==1.7.2
xgboost==3.1.2
transformers==4.57.3
torch==2.9.1
tensorflow==2.21.0
tf_keras==2.21.0

----------------------------------------
COMPATIBILITY STATUS

✔ All above versions are tested together and working in this project.

✔ numpy 1.26.4 works correctly with pandas, scikit-learn, and xgboost.

✔ transformers works with torch (PyTorch backend).

⚠ tensorflow + keras conflict is avoided using:

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

✔ tf_keras is used instead of keras for compatibility.

----------------------------------------
KNOWN INCOMPATIBLE CASES

❌ Changing numpy version after creating .pkl file will cause errors
   (Example: "No module named numpy._core")

❌ Using keras (version 3.x) with transformers may cause runtime errors

❌ Mixing different environments (installing packages globally and in venv) can break execution

❌ Upgrading libraries randomly may cause dependency conflicts

----------------------------------------
HOW TO FIX ISSUES (PROCEDURE)

1. Check installed versions:
pip list

2. Check dependency conflicts:
pip check

3. If errors occur, reinstall correct versions:
pip install numpy==1.26.4
pip install pandas==2.3.3
pip install scikit-learn==1.7.2
pip install xgboost==3.1.2
pip install transformers==4.57.3
pip install torch==2.9.1

4. If environment is broken:
- Delete old environment
- Create new one
- Reinstall above versions

5. Always keep:
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

----------------------------------------
REQUIRED FILES

emails_features_4_classes.pkl  
bert_embeddings.npy  

----------------------------------------
RUN INSTRUCTIONS

Run:

python train_all_models.py  
python train_model_v5_xgboost.py  

----------------------------------------
MODELS INCLUDED

- Logistic Regression (Hybrid)
- Support Vector Machine (Hybrid)
- Random Forest (Hybrid)
- TF-IDF + LR / SVM / RF
- BoW + LR / SVM
- Hybrid BERT + XGBoost (Proposed Model)

----------------------------------------
IMPORTANT NOTES

- Do NOT change numpy version after generating .pkl files
- Always use same environment
- Use pip check to verify compatibility
- Keep backup of working environment

----------------------------------------
END