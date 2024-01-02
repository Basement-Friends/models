FROM python:3.10.2

# Set the working directory
WORKDIR /web-api/toxic-messages/webapp

# Copy the app.py file into the image
COPY app.py .

# Install dependencies
RUN pip install requests beautifulsoup4 python-dotenv

# Specify the command to run your application
CMD ["python", "app.py"]
