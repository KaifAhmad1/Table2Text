# Table2Text
Table2Text is a Streamlit application powered by Open Source techstack `Mistral-8x7b-32k` language model by Mistral AI and the Langchain framework. It assists with queries related to tabular data, such as CSV or Pandas DataFrame, which consist of textual and numerical information.
### **Features and Usage:** 
  - **Upload CSV Files:**
    - Easily upload your CSV files for analysis.
    - Use the file uploader in the sidebar to upload your CSV file.
  - **Data Preview:**
    - Preview and filter your data directly in the app.
    - View and filter your data in the `Data Preview` tab.
    - Select columns and filter values to customize the data displayed.
    - Download the filtered data if needed.
  - **Natural Language Queries:**
    - Ask questions about your data and get instant real-time responses using Mistral AI and Groq.
    - Navigate to the `Query` tab to ask questions about your data.
    - Enter your query in the text input field and get instant responses.
    - Follow up with additional questions to refine your analysis.
  - **Data Exploration:**
    - Visualize your data with scatter plots, bar charts, and heatmaps.
    - In the `Data Exploration` tab, visualize your data using various charts:
     - Statistical Summary
     - Scatter Plot
     - Bar Chart
     - Heatmap
  
``` 
Table2Text/
│
├── app/
│ ├── init.py 
│ ├── model.py 
│ └── config.py 
│
├── app.py 
│
├── README.md 
│
└── requirements.txt 
``` 

## Detailed Explanation of Project Files

- ### **`app.py`**
This is the main script that runs the Streamlit application. It sets up the web interface, handles file uploads, and interacts with the chatbot model to process queries.

- ### **`config.py`**
This file contains configuration parameters for the model, such as the model ID, maximum output tokens, and the Groq API key.

- ### **`model.py`**
This script handles the creation and interaction with the Langchain model. It includes functions to initialize the model and define the chatbot's behavior.

- ### **`requirements.txt`**
  This file lists all the Python dependencies required for the project. Use `pip install -r requirements.txt` to install these dependencies.

- ### **`__init__.py`**
  This file can be empty or used to initialize the package.


- ### **How to run on Windows Locally:**
  - 1. **Clone the Repository:** Clone the repository to your local machine using Git
         ``` sh
         git clone https://github.com/KaifAhmad1/Table2Text.git
         ```
  - 2. **Navigate to the Project Directory:** Navigate to the directory where you cloned the repository
         ``` sh
         cd Table2Text
         ```
  - 3. **Install Dependencies:** Install the required Python packages listed in `requirements.txt` using pip
         ``` sh
         pip install -r requirements.txt
         ```
  - 4. **Run the Streamlit Application:** Run the Streamlit application by executing the `app.py` script
        ``` sh
        streamlit run app.py
        ```
## Acknowledgements: 
- [Langchain](https://github.com/langchain-ai/langchain)
- [Streamlit](https://github.com/streamlit/streamlit)
- [Mistral AI](https://github.com/mistralai)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Groq](https://github.com/groq)
  
         

