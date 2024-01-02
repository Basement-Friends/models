FROM python:3.10.2
# Or any preferred Python version.
ADD app.py .
RUN pip install requests beautifulsoup4 python-dotenv 
CMD [“python”, “./web-api/toxic-messages/webapp/app.py”] 
# Or enter the name of your unique directory and parameter set.