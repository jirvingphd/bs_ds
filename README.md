---
layout: post
title:      "I published my own Python package and you can too! "
date:       2019-05-20 18:19:59 -0400
permalink:  i_published_my_own_python_package_and_you_can_too
---

## *A.K.A. Professional-quality Python packages made easy with cookiecutter:*


> <img src="https://raw.githubusercontent.com/jirvingphd/dsc-3-final-project-online-ds-ft-021119/master/blog_post/cookiecutter_medium.png" width="200"><br>
> <img src = "https://raw.githubusercontent.com/jirvingphd/dsc-3-final-project-online-ds-ft-021119/master/blog_post/cookie%20cutter%20-%20cutting%20code.png" width="150">

<img src="https://raw.githubusercontent.com/jirvingphd/dsc-3-final-project-online-ds-ft-021119/master/blog_post/big%20blog%20header_728%20x%2090.jpg" width="400">

### *How I created BroadSteel_DataScience (bs_ds)*‡


> <img src="https://raw.githubusercontent.com/jirvingphd/dsc-3-final-project-online-ds-ft-021119/master/blog_post/bs_ds_logo.png" width ="200">

- [https://bs-ds.readthedocs.io/en/latest/](https://bs-ds.readthedocs.io/en/latest/)<br>
- [https://pypi.org/project/bs-ds/](https://pypi.org/project/bs-ds/)<br>

‡ *Note: BroadSteel DataScience is: inspired by, an homage to, legally distinct from, and in no way representative of <br>Flatiron School's Online [Data Science bootcamp program.](https://flatironschool.com/career-courses/data-science-bootcamp/online/)* <br> 


___

# In this post, we will:<br>
### 1. **Discuss why you would want to create and publish your own Python package.** 
### 2. **Review the tools I use for editing and publishing bs_ds**
### 3. **Provide step-by-step instructions for using cookiecutter to create professional infrastructure for your package.**
- **Including [auto-generated documentation with Sphinx](https://bs-ds.readthedocs.io/en/latest/)**
- **and automated build-testing and deployment with [travis-ci.org.](https://travis-ci.org/)**
<br>

___

# 1. "But...*why would I bother?*", you may be wondering...
<br>

## **Lots of reasons!** (*mostly convenience*) 
### But also...
1. **Becuase you're *sick* of having to copy and paste your favorite functions into every new notebook.**
    - I work on several computers and also in the cloud. It was always a pain to have to log into Dropbox on any new computer and remmeber where I saved that notebook that had all of those cool little functions I wrote...I 

2. **Because you collaborate with others and want to ensure you all have the same toolkit at your disposal.**
    - My collaborator and fellow student, [Michael Moravetz](https://github.com/MichaelMoravetz), and I have been working closely together and we wanted an easy way to share the cool/helpful tools each of us that we either wrote ourselves or were given in our bootcamp lessons. 

3. **Because you're instructor tells you that your notebook has WAY too many functions up front, and its distracting...**
    - ***You:*** "But aren't I *supposed* to write a lot of functions to become a better programmer?""
    - ***Me:*** "Yes...you're right... BUT when someone has to scroll through a lot of functions-as-appetizers before geting to your main-course-notebook, they may not have much attentional-appetite left."
    
4. **Because you're a little OCD and like having ALL of your notebook's settings JUST THE RIGHT WAY.**
    - pandas.set_options()
    - HTML/CSS styling, etc.)
    - Matplotlib Params

5. **Finally, becuase you're lazy** and just want all of your tools imported for you and ready to use whereever you go with as little effort as possible.
> **I'm a big believer in exerting extra up-front-effort in the name of future-laziness-convience.**


___

# 2. "OK, I get it," you're (hopefully) thinking... "So, *how?*"
* Please note: there are simpler ways to go about creating your first Python module, but I am going to show you the exact tools and instructions I used to allow me to have a totally automated workflow and package deployment.
* 
## THE TOOLS I USE:
- [Anaconda-installed Python](https://www.anaconda.com/)
- [Microsoft's Visual Studio Code](https://code.visualstudio.com/docs/introvideos/basics)
    - Its optionally installed with Anaconda and has quickly become my favorite editor.   *(Sorry, SublimeText3... It's not you! Its me.<br>  I still love you..I'm just not as ***in love*** with you as I used to be. But we can still be friends, right?!)*
- [GitHub Desktop](https://desktop.github.com/)
    - Its SO convenient and simplifies the workflow and collaboration immensely.
-  [Cookiecutter](https://cookiecutter-pypackage.readthedocs.io/en/latest/) python package
    - For the total package infrastructure, with the opton of setting up automation and documentation generation. 
- [Google Colab]( ), [Microsoft Azure Notebooks]( ) 
    - For testing the package in cloud Jupyter notebooks. Just add an '!' to the pip install command in any cloud-notebook <br> ```!pip install bs_ds```


# 3. HOW TO CREATE A PROFESSIONAL INFRASTRUCTURE USING COOKIECUTTER
> <img src="https://raw.githubusercontent.com/jirvingphd/dsc-3-final-project-online-ds-ft-021119/master/blog_post/cookiecutter_medium.png" width="200"><br>

### PLEASE OPEN THE OFFICIAL COOKIECUTTER TUTORIAL:
- [ ] **Open Cookiecutter's [official tutorial from their documentation.](https://cookiecutter-pypackage.readthedocs.io/en/latest/tutorial.html)**
    - Its very good overall but I suggest doing a few steps a little differently. My suggested steps will start with "REC'D" to indicate deviations from the official steps.
- [ ] **Make sure to reference BOTH my instructions beklow as well as the official tutorial.**
    - Make **read my recommended steps first,** as I suggest doing some steps earlier than the tutorial does. 

___
## MY RECOMMENDED STEPS FOR CUTTING YOUR FIRST COOKIES:

### BEFORE GETTING STARTED,  CREATE ACCOUNTS FOR REQUIRED/DESIRED SERVICES:<br>
 1. [ ]  **Go to PyPi.org and "Register" a new account. [REQUIRED]** <br>PyPi.org is the offical service that hosts pip-installable packages. 
    - You will need your account name very soon, during the initial cookie-cutting step below. 
    - You will need your password later,  when you are ready to upload your package to PyPi. <br><br>
    
2. [ ] **Register your github account with www.travis-ci.org.   [OPTIONAL, BUT HIGHLY RECOMMENDED]**<br> for automated build testing and auto-deployment to PyPi.
    - NOTE: Make sure to go to **www.travis-ci.ORG** (**NOT** travis-ci.**COM**, which is for commerical, non-open source pcakges) <br><br>
    - **Suggestion: This is totally worth it.**  It adds a little complexity to the cookie-cutting set up process, but  it:
        - makes updating your package a breeze.
        - makes it easier for others to contribute to your package 
        -  it will pre-test any Pull Requests so you will already know if the code is functional before you merge their code with yours.<br><br>
3. [ ] **Register an account on [readthedocs.org](https://readthedocs.org/) [OPTIONAL, BUT REC'D IF SHARING YOUR WORK]**
    - Readthedocs will host your generated user documentation for your package.
    - Note: Cookiecutter will fill in a lot of the documentation basics for you.
    - Note: There is an additional advanced method to auto-generate all documentation from docstrings, which I will mention in the tutorial below.

___

### **REC'D STEP 0. *BEFORE INSTALING ANYTHING*, you should:**<br>

- [ ] **Create a new virtual environment**, preferably by cloning your current one.
    - Anaconda Navigator makes the cloning process easy. 
        - In Navigator, click on the ```Environments``` tab in the sidebar. 
        - Click on your current enivornment of choice, then click the Clone button, and give it a new name. 
    - [ ] Add this to your Jupyter notebook kernels using <br>```python -m ipykernel install --user --name myenv```
- [ ] **Backup and export your current enviornment** to a .yml file, which you can use to re-install your env, if need be. 
    - For Anaconda environments, open your terminal and activate your environment before exporting: <br><br>```source activate env-name```<br> ```conda env export > my_environment.yml``` <br><br>
        - *where "env-name" is the name of the environment you'd like to clone and "my_environment".yml is any-name-you'd-like.yml*
    - This will save the .yml into your current  directory that can be used to install your environment in the future using:<br>```conda env create -f my_environment.yml```
- **DO NOT SKIP THIS STEP.** I have warned you and I am not responsible for any broken environments.<br>While nothing *should* break, it's always a GOOD idea to create a new environment for creating and installing test packages. Really, I should say  its a DUMB idea not to. <br><br><br>
  
### **REC'D STEP 1: Install cookiecutter into your new environment.**
- Tutorial "Step 1: Install Cookiecutter": Install Cookie Cutter and cookie-cut the default template cookiecuter repo.
    - [x] You may ignore the first part of Step 1 (using virtualenv to create an env). <br><br>
    - [ ] Install cookiecutter via pip:<br>
    ```pip install cookiecutter```<br><br>
    
### **REC'D STEP 2: Create a new GitHub repo for your package and clone it locally**
***NOTE: My recommendation deviates from the tutorial. This will replace "Step 3: Create a GitHub Repo".***
- [ ] Log into your your GitHub profile on github.com and Create a New Repository <br>by clicking the + sign next to your account picture on the top-right of the page. 
    - [ ] Create a New Repository, using the desired name for your published package for the repo name. 
        - [ ] Check "```Initialize this repo with a README```" (you can't clone an empty repo).
            - Leave the rest of the options blank/none.  ```Initialzie with Add a .git ignore```, ```Add a license```
                - Cookiecutter will ask you to choose a license later in the process. 
- [ ] Clone the new repo to your computer. (This is the perfect chance to try using [GitHub Desktop](https://desktop.github.com/), if you haven't before. )
    - Click Clone or Download and:
        -  Copy the url if you plan on using your terminal to clone. 
        - OR "Open in Desktop" if you've installed and logged in to the GitHub Desktop App. <br><br>


### **OFFICIAL STEP #3.  Use the cookiecutter command to cut-your-first-cookie-template.**

* [ ] Activate cloned environment from step #1, ```cd```into your  repo's folder.
* [ ] Enter the following command to create the template infrastructure:

 > ```cookiecutter https://github.com/audreyr/cookiecutter-pypackage.git```<br>

### Cookiecutter Prompts and recommended answers:
- **Cookiecutter will ask you several questions during the cookie-cutting process, [check this resouce to see the descriptions for each prompt.](https://cookiecutter-pypackage.readthedocs.io/en/latest/prompts.html)**

#### The cookiecutter options I selected for bs_ds were as follows:
- **"project_slug"** 
    - should match the name of your new repo from step #2.
    - It should be something terminal-syntax (no -'s or spaces, etc.)
- **"project_name"**
    - will be what appears in all of the generated documentation. It can have spaces and any characters that you wish. 
- **"use_pytest"**:
    - use default 'n'
- **"use_pypi_deployment_with_travis"**:
    - use 'y' for auto-deployment with travis-ci.org (will need an account, as described above)
- **"add_pyup_badge"**:
    - use default 'n'
- **"Select command_line_interface:"**
    - I suggest option 2 for No command-line interface. 
- **"Select open source license"**
    - This is an important choice that determines what people are allowed to do with your code with or without your permission. 
        - Consult https://choosealicense.com/ (github website explaining licenses) for information.
    - Note: bs_ds is published using option 5 - GNU General Public License v3, which choosealicense.com defines as:
>"The GNU GPLv3 also lets people do almost anything they want with your project, except to distribute closed source versions."

- **Cookiecutter will then create a new folder  inside of your main repo folder, whose name is determined by the "project_slug" entered above.**

<br><br>
    
####  **STEP #3B [REQUIRED if you followed REC'D STEP #2 and created the repo first]:**

- If you followed my REC'D STEP #2, you main repo folder should now contain:
    - a README.md file
    - a ".git" folder
    - a new subfolder whose name == project_slug entered above (I will refer to as "slug folder #1")
    
        - Inside of the project_slug folder, you should find:
            - a " .github" folder
            - a "docs" folder
            - a "tests" folder 
            - ANOTHER folder whose name == project_slug (I will refer to as "slug folder #2")
            - and a text file  called "requirements_dev.txt",  several .rst files, setup.py, setup.cfg, and several other files.
            
    - [ ] **Move(or cut) all of the contents from inside slug folder #1 and move/paste them into the main repo folder.**

    - After moving the contents to the main repo folder,  there should be :
        - A project_slug folder (which is actually slug_folder#2 now),
        - requirements-dev.txt 
        - and the .rst and setup files originally from slug folder#2.
    
        - Inisde of the project_slug folder, there should only be 2 files and 0 folders:
            - __init__.py
            - project_slug.py

- **If so congratulations! You have the infrastructure properly installed!** 
    

### **Official Step 4: Install Dev Requirements**
- In your terminal, make sure you are still located in the main repo folder, which contains **requirements-dev.txt**
- Make sure you are still using your newly cloned environment, then enter:<br><br>
```pip install -r requirements_dev.txt```

___

> - [ ] This is a decent place to take a moment to commit your changes and push to  your github repo. 

___

### **Official Step 5: Set Up Travis CI**
- [ ] **In order to follow the offical step 5, you will need to install Travis CLI tool, which requires Ruby.**<br> [Instructions are located here and are OS-specific](https://cookiecutter-pypackage.readthedocs.io/en/latest/travis_pypi_setup.html#travis-pypi-setup), 
    - For MacOS, they recommend using the Homebrew travis package:
        - ```brew install travis```<br><br>
    - For windows, you will need to install ruby and then use ```gem install``` to install travis.
        - [ ] [Install Ruby](http://www.ruby-lang.org/en/downloads/) (if not already installed on your system)
        - [ ] Install Travis CLI tool: (See the OS-specifc instructions directly above)
            - After Ruby is installed, enter the following command to install Travis CLI tool. <br> ```gem install travis -v 1.8.10 --no-rdoc --no-ri```<br><br>
        
- [ ] **Once Travis CLI is installed,  you may continue to follow the [official tutorial instructions for step #5](https://cookiecutter-pypackage.readthedocs.io/en/latest/tutorial.html#step-5-set-up-travis-ci)**
    - NOTE: Here is where you will need to have your password for PyPi available. 
        - **CAUTION:**<br> When entering your PyPi username and  password in the terminal, there will be **NO VISUAL INDICATOR that you have typed your password.** <br><br>
        - There are no characters displayed and no dots or placeholders to indicated the # of characters entered, so **carefully enter your password when prompted and press enter.**<br><br>
    - **TROUBLESHOOTING NOTE: If Travis doesn't does not ask for your password after entering username:**
        - I experienced an issue when attempting to follow step #5, after entering the ```travis encrypt --add deploy.password``` command, you should first be prompted for your username, then your password. 
        - I use Git Bash for my main terminal on Windows and for some reason **Travis would hang after I entered my username and would never ask me for password.**
            - I got around the issue by **using the normal windows cmd prompt for this step instead of using GitBash.**  (This is a one-time step that will encrypt your password and store it in a config file so you never have to enter it again.)


### **Official Step 6: Set Up ReadTheDocs**
- [ ] Follow the [official tutorial step 6](https://cookiecutter-pypackage.readthedocs.io/en/latest/tutorial.html#step-6-set-up-readthedocs) for setting up documentation on readthedocs.org.

### ~~Official Step 7:Set Up pyup.io~~
- **Short Version: This is an added level of complexity that I chose to skip for myself and recommend you do the same for now.**
- I recommended skipping setting up pyup.io during the cookiecutter prompt responses above. 
    - This service would alert you when any of the required python packages that are your package needs to run have been updated, so that you can update the versions in your installation requirements  
___


> **SIDEBAR: As of now, you may realize that you have not actually added any code to your python package, and yet the next official step is to release on PyPi.**

> - If you'd like to add some of your code before submitting your package to PyPi, jump down to the "Adding Your Code / Editing your package" section ( after the official instructions).

___

### **Official Step 8: Release on PyPi**
#### One last annoying, first-time-only hurdle and then you're on your way to automated deployment for the future!
- Travis-CI will automate the process for generating the distribution files for your package and uploading them to PyPi, BUT it cannot CREATE a NEW package that doesn't already exist on PyPi's servers.
- [ ] To *register* your new package with PyPi for the very first version, you must **manually create and upload the very first version** of your package.  [Official Python instructions for "generating distribution archives", summarized below](https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives)
    - Briefly, from inside the main folder of your repo (that contains the setup.py file):
        1. [ ] In your terminal (in your cloned environment), make sure you have the current setuptools installed:<br>
        ```python3 -m pip install --user --upgrade setuptools wheel```
        2. [ ] Build the current version of your package 
        ```python3 setup.py sdist bdist_wheel```
        3. [ ] Install the tool for uploading to pypi, twine:
            ```python3 -m pip install --user --upgrade twine```
        4. [ ] Upload the distribution files created above (inside a new folder called dist/)
        ```twine upload dist/*```
            - [ ] When prompted, enter your PyPi.org username and password. 
        6. Thats it! You can go to PyPi.org, log into your account and you should see your package appear under "Your Projects"  
        - **After a couple moments, your package should be available on pip.<br>
        ```pip install my_package_name``` to install locally or <br>
        ```!pip install my_package_name``` to install in a cloud notebook. 

        - TROUBLESHOOTING NOTE: 
            - For me, using ```python3``` for the above commnads did not work. I simply had to change ```python3``` to just  ```python```<br>
                 - Example:<br>
                   ```python -m pip install --user --upgrade setuptools wheel```
            - If this doesn't fix it for you, you may need to update your systems Path variable (basically a list that tells your computer all of the locations on your PC where you may have scripts/functions saved to run from your terminal).
                - For Windows, [check this article](https://geek-university.com/python/add-python-to-the-windows-path/) for instructions on how to add python to your system path. 
                - For Mac, [try this article's suggestions](https://programwithus.com/learn-to-code/install-python3-mac/)


___

# Adding Your Code / Editing your package/modules

- When working on your package/modules, I highly recommend using **Microsoft Visual Studio Code.**
    - Visual Studio was likely installed with Anaconda, but if it wasn't. Open Anaconda Navigator, and look for Visual Studio code on the Home tab, in the same section as Jupyter Lab and Jupyter Notebooks.

- **The easiest way to manage all of your package's setup files and modules is to the the File > Open Folder option and select your repo's main folder.**

## Important Files You Will Want to Edit:
### The files we will discuss below:
-  **Module/submodule files (where your put your code)**
     -  init.py
     -  module.py
-  **Package creation/installation requirement settings:**
     -  setup.py
- **Documentation creation settings:**
     -  conf.py
- **Versioning and Automated deployment**
     - setup.cfg
     - travis.yml


#### Your package __init__ and module .py files:
- Inside of your main repo folder, you should have your project_slug folder (where project_slug = your package's name)
- **There should be 2 files inside that folder:  __init__.py, and project_slug.py**
     - **init.py is the most critical file of your package.**  When you import your package, you are actually running the init.py file and importing the functions inside it.
        - **The simplest way to add your own functions is to add them to the __init__.py file.**
            - When you use ```import package_name```:
                - The functions and commands contained in your __init__.py file will be imported under your package's name. 
                - Example:<br>```package_name.some_function()```
            -  As with all python packages, you can assign it a short handle to make accessing your functions less tedious:
                - Example<br>  ```import package_name as pn
                pn.some_function()```
                    
            - If you use ```from package_name import *```:
                - All of the functions inside of the init file will be available without needing to specify the package.  

                - Example:<br>
                ```from package_name import *```<br>```some_function()```
        - **The more advanced way to add your own functions is to add them as a sub-module.**
            - The project_slug.py file is actually a submodule of your package, but shares the same name.
                - For bs_ds, we have many functions stored inside of the package submodule:
                    - Which is accessed by bs_ds.bs_ds which is the (package_name).(submodule_name)
                    - The package name is essentially the project_slug folder and then the submodule name is specifying which .py file (INSIDE of that folder) should be imported.
            - See the screenshot below of bs_ds's init file and how it imports submodules. 
   <img src="
        
#### Setup.py
- [ ]  Adding dependencies to be installed with your package:
    - At the top of the file, you will see an empty list called requirements<br>
    ```requirements = [ ]```
    - Add any packages that you would like to be installed with your package. 
        -  If the user is missing any of these pip will install them as well. <br>
    ```requirements = ['numpy','pandas','scikit-learn','matplotlib','scipy','pprint']```
		
#### Controlling Documentation Generation - conf.py
- Documentation generation is done using [Sphinx](http://www.sphinx-doc.org/en/master/index.html)

- **conf.py** controls the settings for the creation of your documentation.
     - [Sphinx's help page on conf.py](http://www.sphinx-doc.org/en/master/usage/quickstart.html#basic-configuration)

- [Read how to create documentation from your functions' docstrings using "sphinx.ext.autodoc" works](http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) 
     - Add this function to the end of conf.py for auto-generated of help from docstrings. 

```python
def run_apidoc(_):
	from sphinx.ext.apidoc import main
	import os
	import sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
	cur_dir = os.path.abspath(os.path.dirname(__file__))
	module = os.path.join(cur_dir,"..","bs_ds")
	main(['-o', cur_dir, module,'-M','--force'])

def setup(app):
	app.connect('builder-inited', run_apidoc)
```

### travis.yml
**travis.yml controls the build testing and deployment process.**
- At the top of the file, there is a list of python versions (3.6, 3.5, etc.)
    - [ ] You may want to remove versions of python that your package cannot support. 
        - For example, f-string formatting wasn't added until Python 3.6 <br> ```
        print(f"Print the {variable_contents}')```
    - Otherwise, your build will fail when travis tests the older version of python, since you used functions that were not compatible with old versions.
    - bs_ds only supports 3.6 at the moment. 
- At the bottom of the file, there is a ```deploy:``` section. 
    - I personally had difficult using ```--tags``` in order to trigger the deployment of bs_ds.
    - I removed the ```tags:true``` line under `on:`, which is at the bottom of the ```deploy:``` section.
    

### setup.cfg

- [ ] If you removed the ```tags:true``` line from travis.yml, you should also remove:
```tag = True```under [bumpversion]
```python
[bumpversion]
current_version = 0.1.0
commit = True
tag = True
 ```
    - This means that instead of waiting for a special --tagged commit to initiate build testing, doc generation, and deployment, the process will be triggered by any commit.
[!]
___

# Updating and Deploying Your Package

## To deploy an updated version of your package:
1. [ ] Debug your modules locally to save time.
2. [ ] Save all updated files and commit them to your repo. 
3. [ ] Bump the version number and commit again. 
4. [ ] Push the repo back to git.
5. [ ] Check travis-ci.org and readthedocs.org for the package and documentation build test results. 

- For the official cookiecutter checklist of steps to deploy an updated version of your package, [see this file.](https://gist.github.com/audreyr/5990987)

### 1. Debug your modules before committing (to save time instead of waiting for Travis-CI)
- Visual Studio Code has a very handy Debug feature, which you can access from the sidebar (its the symbol with the bug on it). 
- On the top of the sidebar that appears, there is a dropdown menu with a green play button. 
    - Open the file you want to test (testing __init__.py is always recommended, but you should test any modules that have been updated. 
    - From this menu, select Python Module.

### 2. Make sure you've saved all changes to the finalized files.


### 3. Increment the version number for your package.
- PyPi.org will only accept new versions of your package if it has a unique version number.
     - It does not matter if your code has changed, PyPi will not publish it if the version number already exists.

#### Using *bumpversion* to increment version #:
- The version number for your package is located in 3 file locations:
    - setup.cfg
    - setup.py
    - init.py

-  **bumpversion** will increment all 3 locations when you enter a bumpversion command in your terminal.
    - bumpversion has understands 3 types of updates: major, minor, and patch.
        - For example, let's say your package is currently v 0.1.0
            - ```bumpversion major``` 
                - Increment version #'s by 1's 
                - v 0.1.0 is bumped to v 1.0.0<br>
            - ```bumpversion minor```
                - increments version by 0.1
                - v 0.1.0 is bumped to v 0.2.0<br>
            - ```bumpversion patch```
                - increments version by 0.0.1
                - v 0.1.0 is bumped to v 0.1.1<br>
    - **Before entering the bumpversion command, you must commit any changes you've made to your repo.**
        - bumpversion will return an error if you try to bump without committing first.
    - [ ] To increment your package's version #: 
        - Commit any changes you've made for your new version. (note: you do not need to ```git push``` yet. Committing the changes is sufficient to appease bumpversion).
    - [ ]  Enter the appropriate bumpversion command depending on how much you'd like to increase the version #.
    - [ ] Push the updated repo. If you removed the tags:true entries as suggested above, Travis-CI will automatically build test and attempt to deploy any commits to your package.

- ***NOTE: While this may sound risky, its actually not, since PyPi will not deploy any packages with the same version number.*** 
    -  As long as you do not bumpversion, Travis will  _test_ your updated package, but it will _fail to deploy it_, since PyPi already has a pre-existing distribution for that version***
   - Documentation Note:
        - Readthedocs.org will test and update your documentation for ANY commit. So if you only need to update an aspect of the doc's, you can simply change the settings and push your repo _without_ having to bumpversion.

### 4. Commit all changes and push your repo back to git

- As long as your removed the suggested edits regardings "tags" described above, your commit will automatically be send to travis-ci and readthedocs for build testing and deployment. 


### 5. To check on the status of your package:
- [  ] **Log into travis-ci.org to see the latest build results and any errors that were found during testing.**<br>
     - If you spent time to debug locally first, you won't be needing the error log nearly as much.  
     - NOTE: Travis-CI will indicate an failed build if it cannot be deployed to PyPi (usually because you did not increment the version #), even if the package itself is fine. Look at the build log to see the error code to determine if this is the case.  <br><br>
- [ ] **Checking documentation creation on readthedocs.org** <br>
     - Log into your accout on readthedocs.org, click on your package and then the green "View Docs" button.
     - Note: Even if Travis fails to deploy your package update to PyPi, readthedocs will still generate new documentation. 

## That's it! (for now)

<img src=https://raw.githubusercontent.com/jirvingphd/dsc-3-final-project-online-ds-ft-021119/master/blog_post/celebrate-311709_1280.png width="400">
# **Congratulations!!!** You did it!

- While the process certainly is not _easy_ to set up, once everything has been configured you will be able to easily and automatically deploy all updates to your package/modules.

- There was more I originally wished to describe, such as detailed explanation of controlling your documentation structure, and setting up collaboration with others, but that will have to wait until next time. 

- If you have any questions please feel free to email me at james.irving.phd@outlook.com and I will help you resolve what I can or at least be able to point you in the right direction to find an answer (if I don't know it myself).



