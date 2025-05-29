The skeleton pipeline is designed to:

- Maximize separation of concerns, accommodating four team members.
- Provide high flexibility for independent or collaborative work without disrupting the workflow.
- Remain lean and lightweight, allowing for easy assembly and disassembly.

Python version used: 3.13.2

The virtual environment is pushed because it contains only five packages: `group4venv`. Will eventually .gitignore afterwards

Alternatively, just run (you probably already have these packages):

```bash
pip install -r requirements/requirements.txt
```

To execute the pipeline, use the following commands:

```bash
python interface.py detect run
python interface.py query ask --prompt "This is a test prompt?"
```

Suggestion steps:
1. Look at interface.py
2. Look at working_file_example.ipynb
3. Look at /entrypoints
4. Look at /group4py/src (custom library)