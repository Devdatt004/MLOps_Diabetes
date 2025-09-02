# Use Python
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .
COPY app/ ./app/
COPY diabetes.csv .
COPY train.py .

# Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
RUN python train.py

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["python", "app/app.py"]

