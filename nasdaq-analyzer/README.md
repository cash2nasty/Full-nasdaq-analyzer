# Nasdaq Analyzer

## Overview
The Nasdaq Analyzer is a Streamlit application designed to analyze historical patterns in the Nasdaq index. It provides users with tools to visualize data, assess market bias, and manage notes related to specific trading days.

## Project Structure
```
nasdaq-analyzer
├── src
│   ├── nasdaq_analyzer3.py       # Main Streamlit application logic
│   ├── data_fetch.py              # Functions for fetching data from various sources
│   ├── indicators.py               # Functions for calculating financial indicators
│   ├── bias_engine.py              # Functions for adding bias columns and predicting market bias
│   ├── notes.py                    # Manages user notes for specific dates
│   └── utils.py                    # Utility functions for various tasks
├── data
│   └── nasdaq_close_history.csv    # Historical closing prices for the Nasdaq index
├── tests
│   └── test_app.py                 # Unit tests for the application
├── .env.example                     # Template for environment variables
├── .gitignore                       # Files and directories to ignore by Git
├── requirements.txt                 # List of required Python packages
└── README.md                        # Documentation for the project
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd nasdaq-analyzer
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables by copying `.env.example` to `.env` and filling in the necessary values.

## Usage
To run the application, execute the following command:
```
streamlit run src/nasdaq_analyzer3.py
```

Once the application is running, you can interact with the dropdown menu to access various functionalities, including:
- Analyzing historical patterns
- Viewing news and events
- Managing notes for specific dates

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.