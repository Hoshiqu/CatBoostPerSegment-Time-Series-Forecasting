README.txt

How to Build My App Using PyInstaller:

1. Create new invoirment 

2. copy requirements.txt, app.py, run.py, dist, hooks

3. Install the required packages by running the following command: 

   pip install -r requirements.txt

4. run the following command:

   pyinstaller --onefile --additional-hooks-dir=./hooks app.py --clean

5. Replace created `app.spec` file with provided and replace all instances of "Streamlit/" 

   with your `<venv-name>`.

6. Build the app using PyInstaller with the following command:

   pyinstaller run_app.spec --clean

7. Once the build process is complete, you will find the executable file (`app.exe`) in the `/dist` folder.

8. Copy app.py and run.py to /dist and start .exe

Note: Make sure to replace `<venv-name>` with the name of your virtual environment.


Here is general guide to
Building the App with PyInstaller:

Streamlit Application Setup and Build Instructions

This README provides instructions on how to install, run, and build a Streamlit application using PyInstaller.

Prerequisites
- Python: Make sure you have Python installed on your system (version >= 3.10). You can download Python from the official website: https://www.python.org/downloads/

Create a Virtual Environment
1. Open a command prompt or terminal.
2. Create a virtual environment by running the following command:
   python -m venv <venv-name>
3. Activate the virtual environment:
   <venv-name>\Scripts\activate.bat
4. Verify that the virtual environment is active:
   python --version

Installation
1. Open a command prompt or terminal.
2. Install the required packages by running the following command:
   pip install -r requirements.txt

Running the App
1. To run the Streamlit app, navigate to the root folder of your project in the command prompt or terminal.
2. Run the following command:
   streamlit run {app_name}
   Replace {app_name} with the name of your application.

Building the App with PyInstaller
1. Add the main file called run.py to the project.
2. Create an entry point for the executable called app.py.
3. Create a hook file to get Streamlit metadata:
   - Create a new folder called hooks at the root of the project.
   - Inside the hooks folder, create a file named hook-streamlit.py.
   - Add the following code to hook-streamlit.py:
     from PyInstaller.utils.hooks import copy_metadata
     datas = copy_metadata("streamlit")
     datas += copy_metadata("etna")
4. Generate the run_app.spec file by running the following command:
   pyinstaller --onefile --additional-hooks-dir=./hooks app.py --clean
5. Modify the app.spec file:
   - Locate the section starting with a = Analysis(['app.py'],.
   - Add the necessary entries to the datas list.
   - Include the required entries in the hiddenimports list.
   - Set the hookspath accordingly.
6. Load the modifications in the datas by running the following command:
   pyinstaller run_app.spec --clean
7. The build process is complete. Find the run_app.exe file in the dist folder, copy app.py, run.py in /dist
   and run .exe.

Best regards,
Michael