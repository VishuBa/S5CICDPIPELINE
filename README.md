# MNIST ML Project with CI/CD

This project implements a simple DNN for MNIST classification with automated testing and CI/CD pipeline.

## Local Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run tests:
```bash
python -m pytest tests/
```

4. Train the model:
```bash
python src/train.py
```

5. Test the model:
```bash
python src/test.py
```

## Project Structure
- `src/`: Source code
- `tests/`: Test files
- `.github/workflows/`: CI/CD configuration 