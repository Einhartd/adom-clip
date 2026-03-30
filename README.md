# ADOM 26L - CLIP Project

## Required Dependencies

1. **Conda** - virtual environment manager
    * [Download and install Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)


## Environment Setup

1. Create the base environment using Conda:

```bash
conda env create -f environment.yml
```

2. Activate the environment:

```bash
conda activate adom-clip
```

3. Install the required dependencies:

```bash
uv pip install -r requirements.txt
```

## Verify Setup

To verify that everything works correctly, run the test script:

```bash
python src/test_setup.py
```


