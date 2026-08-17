# ChurnLab

ChurnLab is a machine learning project focused on understanding and implementing Logistic Regression from scratch for customer churn prediction.

The main goal is to learn the mathematics, optimization, preprocessing, and evaluation behind binary classification.

---

# Prerequisites

Both developers should use the same Python version and a current stable version of Git.

## 1. Python 3.12.7

### Download

Download the **Python 3.12.7 Windows 64-bit installer** from the official Python release page:

[Python 3.12.7 — Official Release](https://www.python.org/downloads/release/python-3127/?utm_source=chatgpt.com)

On the download page, under **Files → Windows**, download:

```text
Windows installer (64-bit)
```

### Installation

Run the installer.

On the first installation screen:

**IMPORTANT:** Enable:

```text
Add python.exe to PATH
```

Then select:

```text
Install Now
```

### Verify Python

After installation, close and reopen PowerShell.

Run:

```powershell
python --version
```

Expected:

```text
Python 3.12.7
```

Also verify:

```powershell
py --version
```

If Python 3.12.7 is installed correctly, you should be able to create the project environment using:

```powershell
python -m venv .venv
```

If `python --version` does not show `Python 3.12.7`, stop and resolve the Python installation before continuing.

---

# 2. Git

Git is required for version control and GitHub collaboration.

### Download

Download **Git for Windows** from the official website:

[Git for Windows — Official Download](https://git-scm.com/install/windows?utm_source=chatgpt.com)

Use the **current stable Git for Windows x64 installer**.

The exact Git patch version does not need to be identical between developers. Both developers should use a current stable release.

### Installation

Run the Git installer.

For normal project development, the default installation options are sufficient.

Continue through the installer using the recommended/default settings.

### Verify Git

Close and reopen PowerShell.

Run:

```powershell
git --version
```

You should receive a Git version, for example:

```text
git version 2.55.0
```

The exact version may be newer or slightly different.

---

# Project Setup

## 1. Clone the repository

```powershell
git clone <repository-url>
cd churnlab
```

## 2. Create a virtual environment

Each developer must create their **own local virtual environment**.

```powershell
python -m venv .venv
```

## 3. Activate the virtual environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should show:

```text
(.venv)
```

Verify Python:

```powershell
python --version
```

Expected:

```text
Python 3.12.7
```

Also verify that Python is coming from the project environment:

```powershell
python -c "import sys; print(sys.executable)"
```

The path should end with:

```text
churnlab\.venv\Scripts\python.exe
```

## 4. Install project dependencies

```powershell
pip install -r requirements.txt
```

---

# Git Collaboration

The project uses three levels of branches:

```text
main
  ↑
dev
  ↑
feature branches
```

## `main`

Stable/final version of the project.

Do not work directly on `main`.

## `dev`

Development and integration branch.

Completed feature work is merged into `dev` first.

## `feature/*`

Individual development branches.

Examples:

```text
feature/logistic-regression
feature/preprocessing
feature/metrics
```

---

# Basic Development Workflow

Before starting new work:

```powershell
git checkout dev
git pull origin dev
```

Create a feature branch:

```powershell
git checkout -b feature/your-feature-name
```

Work on your assigned task and test your changes.

Check your changes:

```powershell
git status
```

Stage changes:

```powershell
git add .
```

Commit:

```powershell
git commit -m "Describe your change"
```

Push your feature branch:

```powershell
git push -u origin feature/your-feature-name
```

Create a Pull Request:

```text
feature/your-feature-name → dev
```

After review and testing, merge the Pull Request into `dev`.

The `main` branch is reserved for the stable project version.

---

# Collaboration Rules

* Do not work directly on `main`.
* Do not commit `.venv/`.
* Keep feature branches focused on one task.
* Pull the latest `dev` before starting new work.
* Test your changes before creating a Pull Request.
* Keep commits clear and meaningful.
* Discuss major changes with the other developer before modifying shared work.

---

# Project Structure

```text
churnlab/
│
├── data/
│   └── dataset.csv
│
├── src/
│   ├── preprocessing.py
│   ├── logistic_regression.py
│   ├── metrics.py
│   └── train.py
│
├── visualization/
│   └── plots.py
│
├── app/
│   └── app.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

The project structure may be updated as development progresses.
