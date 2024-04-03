README.txt

Introduction:
This README provides instructions on how to install, run, and build a Streamlit application using Briefcase.

Prerequisites:
Python: Make sure you have Python installed on your system (version >= 3.10). You can download Python from the official website: https://www.python.org/downloads/

Installation:

1. Open a command prompt or terminal.
 
2. Upgrade pip:
   Run the following command to upgrade the packages
   python.exe -m pip install --upgrade pip
3. Install the required packages:
   Run the following command to install the packages specified in requirements.txt:
   pip install -r requirements.txt


Building the App with Briefcase:

1. Initialize a new Briefcase application:
   Run:
   briefcase new
   Then navigate to the created project directory:
   cd streamlitmlapp

2. Setup project files:
   Move __main__.py, app.py, run.py to the src/{app_name}/ directory.
   
   Inject or copy the contents of requirements.txt to pyproject.toml on requires = [], using "" for each.


3. Create your application:
   Run the following command:
   briefcase create

4. Run the app in a development environment:
   Use the following command:
   briefcase dev --no-run

5. Build the app:
   Use the following command:
   briefcase build

6. Run the built app:
   Use the following command:
   briefcase run

7. Package the app:
   Use the following command:
   briefcase package
   This command will generate an .msi installer file for your Streamlit application.

Additional Resources:
For more information and troubleshooting, refer to the Briefcase documentation: 
https://beeware.org/project/projects/tools/briefcase/

Best regards,
Michael