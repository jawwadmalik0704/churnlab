# ChurnLab

ChurnLab is a machine learning project focused on understanding and implementing **Logistic Regression from scratch** for customer churn prediction.

The main goal is to learn the mathematics, optimization, preprocessing, and evaluation behind binary classification.

---

# Prerequisites

Both developers should use the same Python version.

## 1. Python 3.14.7

### Download

Download **Python 3.14.7 for Windows** from the official Python release page:

https://www.python.org/downloads/release/python-3147/

Under **Files**, download:

```text
Windows installer (64-bit)
```

Do **not** use the embeddable/ZIP package for this project.

### Installation

Run the installer.

On the first installation screen, make sure this option is enabled:

```text
Add python.exe to PATH
```

Then continue with the normal installation.

### Verify Python

After installation, close PowerShell and open a **new PowerShell terminal**.

Run:

```powershell
py -3.14 --version
```

Expected:

```text
Python 3.14.7
```

Also run:

```powershell
python --version
```

This should normally show:

```text
Python 3.14.7
```

If `python --version` shows another version, that is not necessarily a problem. For this project, the important command is:

```powershell
py -3.14
```

because it explicitly selects Python 3.14.

---

# 2. Git

Git is required for version control and GitHub collaboration.

### Download

Download **Git for Windows** from the official website:

https://git-scm.com/install/windows

Use the current stable **Git for Windows 64-bit** installer.

The exact Git patch version does not need to be identical between developers.

### Installation

Run the Git installer.

The normal/default installation options are sufficient for this project.

### Verify Git

Close and reopen PowerShell after installation.

Run:

```powershell
git --version
```

Expected output will look similar to:

```text
git version 2.x.x
```

The exact version may be newer.

---

# Project Setup

Each developer must create their **own local virtual environment**.

The `.venv` directory is local to each computer and is **not uploaded to GitHub**.

Both developers must use:

```text
Python 3.14.7
```

---

## 1. Clone the repository

Clone the repository:

```powershell
git clone <repository-url>
```

Enter the project:

```powershell
cd churnlab
```

---

## 2. Create your own virtual environment

Use Python 3.14 explicitly:

```powershell
py -3.14 -m venv .venv
```

This creates:

```text
churnlab/
└── .venv/
```

Each developer creates their own `.venv` on their own computer.

**Do not copy your `.venv` to another developer.**

---

## 3. Activate the virtual environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should show:

```text
(.venv) PS D:\churnlab>
```

The exact path may be different on another computer.

---

## 4. Verify the virtual environment

Run:

```powershell
python --version
```

Expected:

```text
Python 3.14.7
```

Then verify which Python executable is being used:

```powershell
python -c "import sys; print(sys.executable)"
```

Expected:

```text
D:\churnlab\.venv\Scripts\python.exe
```

The path will naturally be different on another developer's computer.

The important part is:

```text
churnlab\.venv\Scripts\python.exe
```

---

## 5. Install project dependencies

Once `requirements.txt` contains the project's dependencies, run:

```powershell
pip install -r requirements.txt
```

Do not install random packages that are not required by the project.

---

# Virtual Environment — Why Do We Use It?

A virtual environment is like a **private Python workspace for one project**.

For example:

```text
Computer
│
├── ChurnLab
│   └── .venv
│
├── Resume Analyzer
│   └── .venv
│
└── Other Project
    └── .venv
```

Each project can have its own packages and package versions without interfering with other projects.

For ChurnLab:

```text
Python 3.14.7
       ↓
     .venv
       ↓
ChurnLab packages
```

Each developer has their **own `.venv`**.

The `.venv` directory should never be committed to Git.

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

The stable/final version of the project.

Do not work directly on `main`.

## `dev`

Development and integration branch.

Completed feature work is merged into `dev` first.

## `feature/*`

Individual development branches.

Examples:

```text
feature/data-loading
feature/logistic-regression
feature/preprocessing
feature/metrics
feature/visualization
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

Work on your assigned task.

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

- Do not work directly on `main`.
- Do not commit `.venv/`.
- Do not commit personal Python installation files.
- Keep feature branches focused on one task.
- Pull the latest `dev` before starting new work.
- Test your changes before creating a Pull Request.
- Keep commits clear and meaningful.
- Discuss major architectural changes before implementing them.
- Do not overwrite another developer's work without discussion.
- Both developers should understand the code they contribute.
- AI may be used for learning, explanation, and debugging, but code should not be blindly copied.

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

---

# Important Python Version Rule

ChurnLab uses:

```text
Python 3.14.7
```

Both developers should create their virtual environment using:

```powershell
py -3.14 -m venv .venv
```

This ensures that both developers use the same Python major/minor version.

The `.venv` directory itself is **local and must not be pushed to GitHub**.
