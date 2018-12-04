# interim_project



1. Install Python from https://www.python.org/downloads/release/python-367/
2. pip3 comes installed with python by default, you must navigate to the directory which Python is installed if it is not in your
   $PATH variable in order to run both python and pip, or use the full path when running these scripts from a different directory

		sudo apt-get install build-essential gfortran libatlas-base-dev python-pip python-dev
		sudo pip install --upgrade pip

	If this does not work, your Ubuntu installation may have its own method of installing scipy, please google to find out.
	
	if on windows install scipy by downloading scripy from http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
	Then use
			pip install <Scipy whl file here>

	Other Requirements
	
   			pip3 install numpy matplotlib keras tensorflow imutils skikit-learn Pillow opencv-python 
  			pip install numpy matplotlib keras tensorflow imutils skikit-learn Pillow opencv-python 


If using custom.py to train, change the variable dataset to the full path of the 'small' directory
If using vgg.py to train, change the variables train_dir and validation_dir to
train_dir - full path to data/tr
validation_dir - full path to data/val

Training scripts are ran using 
	- python train.py 
	or 
	- python vgg.py

Datasets are split using train_test_split.py (Credit: https://gist.github.com/bertcarremans/679624f369ed9270472e37f8333244f5)
Due to the limitations of Github Storage, both datasets for train.py and vgg.py have already been split and uploaded to
https://megaupload.nz/p4o8D1m2b3/Datasets_zip


Django:

In order to install Django, follow the instructions here:
https://docs.djangoproject.com/en/2.1/topics/install/

Copy the 'website' directory to your django_projects folder

To run Django, navigate to the django/django_projects/website, edit the permissions of the fils using:

	chmod -R 777 django/django_projects/website

Then navigate django/django_projects/website and run the command:
	
	If using a public IP: python3 manage.py runserver 0.0.0.0:8000
        If using localhost: python3 manage.py runserver
	
In order to execute the remote script, you will need a remote server hosting a simple hello_world.py in the servers root directory
that prints "Hello World" to the cli.
Then change the IP, username, password in the file website/remotecon.py before running



