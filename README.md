# ml-clean-code: Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Your project description here.

## Files and data description
Overview of the files and data present in the root directory. 

- data/
    - source data
- images/
    - store images
- logs/
    - store logs
- models/
    - store models
- tests/
    - store test results
- churn_notebook.ipynb
    - given: contains the code to be refactored
- churn_library.py
    - _completed:_ define the functions
- churn_script_logging_and_tests.py
    - _completed_: tests and logs
- environment.yml
    - python libraries
- Guide.ipynb
    - given: getting started and troubleshooting
- README.md
    - _completed:_ provides project overview, and instructions to use the code
- requirements_py3.6.txt
- requirements_py3.8.txt
- requirements_py3.10.txt

## Running Files

- Run tests with: 
```Bash
pytest churn_script_logging_and_tests.py
``` 
    - for debugging we can run:
```Bash
pytest churn_script_logging_and_tests.py --pdb
```        
        - to set a breakpoint in the code we can use
        ```Python
        breakpoint()
        ```
    

### Setting up the environment
The file `environment.yaml` was created based on `requirements_py3.10.txt`

Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate custch
```

