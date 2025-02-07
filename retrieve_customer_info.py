import pandas as pd

# Sample customers data (as an example)
data = {
    'name': ['Peak Performance Co.', 'Tech Innovations LLC', 'Green Solutions', 'Creative Minds'],
    'address': ['123 High St', '456 Tech Rd', '789 Green Ave', '101 Innovation Blvd'],
    'email': ['contact@peakperformance.com', 'info@techinnovations.com', 'support@greensolutions.com', 'hello@creativeminds.com'],
}

# Create a DataFrame from the sample data
customers = pd.DataFrame(data)

def retrieve_customer_info(name: str) -> str:
    """Retrieve customer information based on their name."""
    # Filter customers for the customer's name
    customer_info = customers[customers['name'] == name]
    return customer_info.to_string()

# Call the function on "Peak Performance Co."
print(retrieve_customer_info("Peak Performance Co."))
