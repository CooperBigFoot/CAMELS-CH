import requests
import pandas as pd
import io
from datetime import datetime, timedelta
import dotenv
import os
import logging

# Load environment variables from .env file
dotenv.load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGatewayClient:
    def __init__(self, api_key):
        self.base_url = "https://data-gateway.ieasyhydro.org/api"
        self.api_key = api_key

    def get_snow_reanalysis(self, hru_code, start_date=None, end_date=None, param="swe"):
        """
        Retrieve snow reanalysis data for a specific location and parameter.

        :param hru_code: HRU code for the specific location (e.g., "00003" for basin-averaged values)
        :param start_date: Start date for data retrieval (YYYY-MM-DD). Defaults to 30 days before today.
        :param end_date: End date for data retrieval (YYYY-MM-DD)
        :param param: Parameter to retrieve (default: "swe" for Snow Water Equivalent)
        :return: Pandas DataFrame with the CSV data
        """
        # If no start_date provided, default to the last 30 days
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Construct query parameters
        params = {
            "template_type": "RSMinerva",
            "start_date": start_date,
            "hru_code": hru_code,
            "param": param,
            "api_key": self.api_key,
        }

        # Add end_date if provided
        if end_date:
            params["end_date"] = end_date

        try:
            # Make API request
            response = requests.get(
                f"{self.base_url}/calculations/snow-reanalysis/template/RSMinerva",
                params=params,
            )
            response.raise_for_status()

            # Use pandas to read the CSV response content
            df = pd.read_csv(io.StringIO(response.text))
            logger.info(f"Retrieved CSV columns: {df.columns.tolist()}")

            # If necessary, convert the column with SWE data to float (if not already numeric)
            # Uncomment and adjust the column name if needed:
            # df["swe"] = pd.to_numeric(df["swe"], errors="coerce")

            return df

        except requests.RequestException as e:
            logger.error(f"API Request Error: {e}")
            raise


# Example usage
if __name__ == "__main__":
    API_KEY = os.getenv("DATA_GATEWAY_API_KEY")
    if not API_KEY:
        raise ValueError("Please set the DATA_GATEWAY_API_KEY environment variable.")

    client = DataGatewayClient(API_KEY)

    try:
        # Retrieve SWE data for the basin (HRU code "00003")
        df = client.get_snow_reanalysis(
            hru_code="00003",
            start_date="2023-09-01",
            end_date="2023-09-30",
            param="swe"
        )
        print(df)
    except Exception as e:
        print(f"Error: {e}")
