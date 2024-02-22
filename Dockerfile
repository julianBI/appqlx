# Use the latest Ubuntu image as a base
FROM ubuntu:latest

# Set the working directory inside the container
WORKDIR /app

# Update the package list and install Python 3
#RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get update && \
    apt-get install -y python3-pip python3.9 git && \
    apt-get clean

# Install the required Python libraries
#RUN pip3 install pandas numpy matplotlib seaborn 
#RUN pip3 install jupyter
# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Streamlit and Jupyter
# RUN pip install --no-cache-dir streamlit==1.28.0 jupyter
RUN pip install --no-cache-dir streamlit==1.28.0

# Expose the ports for Streamlit and Jupyter
# EXPOSE 8501 8888
EXPOSE 8501

# Create a volume to mount the local directory
VOLUME ["/app"]


# Set the default command to run both Streamlit and Jupyter
# CMD ["sh", "-c", "streamlit run --server.port 8501 /app/my_streamlit_app.py & jupyter lab --port 8888 --no-browser --ip=0.0.0.0 --notebook-dir=/app --allow-root"]
# CMD ["sh", "-c", "streamlit run --server.port 8501 /app/my_streamlit_app.py & jupyter lab --port 8888 --no-browser --ip=0.0.0.0 --notebook-dir=/app --allow-root --NotebookApp.token=''"]
CMD ["sh", "-c", "streamlit run --server.port 8501 /app/my_streamlit_app.py"]

# Install transformers separately without its dependencies
#RUN pip3 install --no-cache-dir transformers



