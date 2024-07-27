# Use the official Python image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Define environment variable
ENV NAME HeartDiseasePrediction

# Run app.py when the container launches
CMD ["python", "src/app.py"]