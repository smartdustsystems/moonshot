FROM tensorflow/tensorflow:latest-jupyter

WORKDIR /tf

# Install additional dependencies
RUN pip install --no-cache-dir \
    spacy \
    beautifulsoup4 \
    requests \
    seaborn \
    scikit-learn \
    pandas \
    numpy \
    tensorflow \
    matplotlib \
    mord \
    xgboost \
    "dask[complete]" \
    pyarrow==10.0.1

# Download Spacy model
RUN python -m spacy download en_core_web_sm

# Expose Jupyter Notebook
EXPOSE 8888

# Default command to run Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
