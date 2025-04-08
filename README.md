# Population Graph Model for Predicting Immunotherapy Clinical Benefits in Non-Small Cell Lung Cancer Patients

## Project Structure

### Step 1. GCN-master 
- Constructs population graphs based on patients' baseline clinical characteristics
- Utilizes 9 baseline features to build patient similarity graphs before immunotherapy

### Step 2. GCN_HGNN.py
- Implements DHGN prognostic model construction and testing
- Generates immunotherapy prognostic scores for patients

## Important Note
Before running the project:
- Update the median survival time in the code with your own dataset values (locations marked with comments)

## Acknowledgments
We gratefully acknowledge the contributions of Huang, Yongxiang and Chung, Albert CS to the PAE model.
