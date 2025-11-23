FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system build deps used by some Python packages (prophet, numpy/scipy builds, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       gfortran \
       libgomp1 \
       libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the Case1 folder into the image so Case2 is not affected
COPY Case1_turnover_outliers/ ./Case1_turnover_outliers/

# Upgrade pip and install Python dependencies used by Case1.
# Note: `prophet` can require additional system deps on some platforms. If the build fails,
# consider using a conda-based image (mambaforge/miniconda) or installing prebuilt wheels.
RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    pandas numpy matplotlib seaborn scikit-learn pyod streamlit prophet==1.1.4 shap || pip install --no-cache-dir pandas numpy matplotlib seaborn scikit-learn pyod streamlit

ENV PATH="/root/.local/bin:${PATH}"

EXPOSE 8501

# Default command: run the Streamlit dashboard inside the Case1 folder
CMD ["streamlit", "run", "Case1_turnover_outliers/dashboard_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
