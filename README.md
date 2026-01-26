# LEAGUE OF LEGENDS SELF-PLAYING AGENT

under development ...
# 1. Installation
1. Download the source code with git
    ```
    git clone https://github.com/sgallis/League-of-Legends-Self-Playing-Agent.git
    ```
2. Create conda environment:
    ```
    conda create -y -n leaguebot python=3.11
    conda activate leaguebot
    ```
3. Install pytorch 2.6.0
    ```
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
    ```
4. Install lightning 2.5.1
    ```
    pip install lightning==2.5.1
    ```
5. Install remaining dependencies:
    ```
    pip install -r requirements.txt
    ```
6. Install game from [League of Legends download link](https://www.leagueoflegends.com/en-gb/download/)
