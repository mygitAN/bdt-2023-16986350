# Use a base Python image
FROM python:3.8-slim

# Set the working directory in the container to /usr/src/app
WORKDIR /usr/src/app

# Copy the Python script into the container
COPY ./bdt-sentiment.py .

# Install the necessary Python libraries
# Update pip
RUN pip install --upgrade pip

# Install the necessary Python libraries
RUN pip install torch torchvision
RUN pip install pandas transformers requests google-cloud-storage scikit-learn


# Make the script executable
RUN chmod +x ./bdt-sentiment.py

# Define the command to run when the container starts
CMD ["python", "./bdt-sentiment.py"]


