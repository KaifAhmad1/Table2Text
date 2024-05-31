# Table2Text
Table2Text is a Streamlit application powered by Llama 3 by Meta and Langchain framework that can assist queries related to Tabular data which consist of textual and mumerical info. Eg Panas Dataframe 
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


- ### **How to run**
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

         

