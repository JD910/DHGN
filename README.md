# Population Graph Model to Predict the Clinical Benefits of Immunotherapy in Patients with Non-small Cell Lung Cancer

## Project Structure

### Step 1. GCN-master 
- Constructs population graph based on the patients' baseline clinical characteristics
- Utilizes nine baseline features to build patient similarity graphs before immunotherapy

### Step 2. GCN_HGNN.py
- Implements DHGN prognostic model construction, training and external test
- Generates prognostic scores for immunotherapy survival outcome prediction in patients with NSCLC

## Important Note
Before running the project:
- Update the median survival time in the code with your own dataset (marked with comments)

## Acknowledgments
We gratefully acknowledge the contributions of Huang, Yongxiang and Chung, Albert CS to the PAE model.
