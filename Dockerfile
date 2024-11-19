FROM bitnami/spark:latest
# Install Python and NumPy
USER root
RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install numpy && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip3 install streamlit
USER 1001
