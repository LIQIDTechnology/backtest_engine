<div align="center"><img src="https://app.liqid.de/build/images/browsers/liqid-logo.png" width="200"></div>
<h3 align="center">Backtest Engine</h3>

<!-- ABOUT THE PROJECT -->
## About The Project



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

#### Install Git
Before getting started make sure you have ```git``` installed by simply running ```git``` from the Terminal.
  ```sh
  git -- version
  ```
If you don’t have it installed already, it will prompt you to install it.

#### Generate an SSH key in macOS
1. Enter the following command in the Terminal window. This starts the key generation process. When you execute this command, the ```ssh-keygen``` utility prompts you to indicate where to store the key.
  ```sh
  ssh-keygen -t rsa
  ```
2. This starts the key generation process. When you execute this command, the ```ssh-keygen``` utility prompts you to indicate where to store the key.
3. Type in a passphrase. You can also hit the ENTER key to accept the default (no passphrase). However, this is not recommended.After you confirm the passphrase, the system generates the key pair. Your public key is saved to the id_rsa.pub.
4. Add your public key to your github account

### Installation


1. Clone the repo. 
   ```sh
   git clone git@github.com:LIQIDTechnology/backtest_engine.git
   ```
2. Create virtual environment in the repository 
   ```sh
   virtualenv venv
   ```
3. Install packages
   ```sh
   pip install -r requirements
   ```
4. Enter your root folder in `config.ini`
   ```js
   root_path = 'ENTER YOUR ROOT PATH TO PROJECT FOLDER'
   ```




