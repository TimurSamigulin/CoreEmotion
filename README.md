<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/TimurSamigulin/core-emotion">
    <img src="docs/image/reaction.png" alt="Logo" width="125" height="125">
  </a>

<h3 align="center">CoreEmotion</h3>

  <p align="center">
    Software module for determining the psycho-emotional state of users by text messages in the Avatar ecosystem
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About

This application is designed to determine the user's psycho-emotional state by text messages exchanged between people in the Avatar ecosystem. Psycho-emotional state data from the Avatar ecosystem is used to personalize artificial lighting when multiple users use luminaires at the same time. This software is part of the avatar ecosystem.


## Documentation
The application documentation is available in the [GitHub wiki](https://github.com/TimurSamigulin/core-emotion/wiki) of this repository. There you can find instructions for installing the application from the source codes, using and configuring the use, as well as all technical information about the project.


## Installation Guide
Clone repository 

    git@github.com:TimurSamigulin/core-emotion.git


Download the emotion detection model from [link](https://drive.google.com/drive/folders/11DcQ9zA4S78VkNVfnheHlF4_6i8iJ0Nd?usp=sharing) and add it to the app/models/emotion-detection folder

To run these software, you need:

install on the system have Docker and the Docker-Compose utility.

run command in terminal

    docker-compose up -d

In case of problems with Docker-Compose installed via the system's package manager, you can install docker-compose in a Python virtual environment:

    pip install docker-compose

In order to run the software locally, you need:

If the software was cloned from the repository, you need to create a virtual environment

    python3.9 -m venv .venv

and install dependencies (command for installation will be added later)

Activate Python virtual environment

    source .venv/bin/activate

Run software

    python -m app

If there are errors with the launch of Redis, mongodb, etc., you need to check for the presence of an .env file and, if it is missing, execute

    sudo cp .env.example .env

If you need to install dependencies for an existing software, then you need to use the command

    pip install -r requirements.txt --extra-index-url http://pypi:zURsnxpa7uge5w4F@5.53.125.17:8080 --trusted-host 5.53.125.17

This is due to the fact that some certain versions of the libraries are on local ITMO servers


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
The application was developed within the framework of the research project "Development of mechanisms for designing the processes of users' vital activity into the ecosystem of their digital assistants" No. 621308


## Developers
Laboratory of Cognitive Non-Verbalics

Samigulin Timur, email: timursamigulin98@gmail.com

<div align="center">
<img src="docs/image/NCCR_LogoBlock_Norma-01_Color-3_NotText_NotSlogan_Eng.png" alt="Logo" width="425">
</div>