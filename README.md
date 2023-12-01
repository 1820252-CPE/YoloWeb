# Brahmmy's Guide

Installation Instructions:

The steps followed by the developers
Step 1: Dataset gathering 
 1. Browsed Google and Roboflow for gathering the images for training the custom data set
 2. Create a main folder. On the main folder of datasets create two sub folders named: images and labels. Inside of those folders create two folders named: train and val.
 3. Storing and sorting the images into two categories: train, and validation with a balance ratio.
 4. Save the images in a folder and separate them into the categories folders.
 5. Annotating the images in makesense.ai or roboflow that can be access on this link: 
      Roboflow: https://roboflow.com/annotate
      Makesense: https://www.makesense.ai/
 6. Save the annotated text files on their respective folders
Step 2: Training the dataset
 1. Save the images and annotations, then compiled it to their respective folders on Google Drive.
 2. Utilizing Google Colab in training the dataset
 3. Configure the Google Colab for training such as pip install PyTorch
 4. Connect the Google Drive to the Google Colab to be trained.
 5. Secure the best.pt model from the Google Colab as the selected PyTorch model checkpoint file
 6. Rename the best.pt model to desired name
Step 3: Web App Development
 1. Install Visual Studio Code as the source code editor that can be accessed from this link: https://code.visualstudio.com/download
 2. Download python 3.11 which is the suitable version that can be accessed from this link: https://www.python.org/downloads/release/python-3115/?fbclid=IwAR1tbaaegVYTdwM2NwQs93c1vqvM4RW-kX7ub8uMBbAry9SHWFFwZwJWsuE
 3. Create a folder for the web app directory like this:
    The folder will contain the app.py for the web app python code and custom dataset.pt and templates folder
    On the templates folder it will contain our html file  named index.html with javascript and css code inside and text file for the Bert Model NLP.
 4. Create the following files app.py for the python file of the web application, index.html for the web structure, design and script, context.txt for the Bert 
    Model, and upload it to the folder. Also include the custom data set according to the provided picture.
Note: bert.py is the code for the Bert Model NLP and main.py is the original code for object detection
 5. Download the python extension on the VSCode and pip install the opencv on VSCode terminal with this code that is essential for running python:
    “pip install opencv-python”
 6. Configure the file main.py python code that served as a basis for the web app that can be accessed from this link: 
     https://drive.google.com/drive/folders/1_q5hMx83DTvd45mIANsxn31E9aKq3YfM
 7. On VS code terminal pip install the following:
pip install Flask
pip install Flask-MySQLdb
pip install ultralytics
pip install transformers
8. Set the best trained model into the app.py web application
9 .Developed python code that contains the functions
10. Developed index.html file accordingly on the desired look to enhance the visual appearance of the web app
11. Using flask deploy the web application on a localhost
Step 4: Database Setup
 1. Download Xampp  for the web development stack that includes Apache, MySQL, PHP that can be accessed from this link:  https://www.apachefriends.org/download.html
 2. On VS code terminal pip install this code:
“pip install Flask-MySQLdb”
 3. Connect the python app into mysql with this code:
app.config['MYSQL_HOST'] = "localhost"
app.config['MYSQL_USER'] = "root"
app.config['MYSQL_PASSWORD'] = ""
app.config['MYSQL_DB'] = "detection"
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

4. Run the application of XAMPP and click start button on Apache and MySQL to access the localhost database that can be accessed from this link: 
http://localhost/phpmyadmin/
5. Create database table
CREATE TABLE IF NOT EXISTS detection_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    object_type VARCHAR(255) NOT NULL,
    count INT DEFAULT 0,
    UNIQUE KEY (object_type)
);

The steps followed by the users

Step 1: Downloading the source codes and editors
1. Install Visual Studio code as the source code editor that can be accessed from this link: https://code.visualstudio.com/download
2. Download python 3.11 which is the suitable version that can be accessed from this link: https://www.python.org/downloads/release/python-3115/?fbclid=IwAR1tbaaegVYTdwM2NwQs93c1vqvM4RW-kX7ub8uMBbAry9SHWFFwZwJWsuE
3. Access the repository and clone the github repository that can be accessed from this link: https://github.com/1820252-CPE/YoloWeb
Step 2: File Management and Installing the requirements
1. Extract the downloaded files on a specific folder on the Desktop
2. Open app.py using VSCode to access the web application
3. Download the Python extension on VS Code
4. On the terminal of VSCode pip install the following code from the pipinstall.txt


