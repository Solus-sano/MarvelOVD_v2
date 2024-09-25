import os
import subprocess


for dir in os.listdir('./'):
    if os.path.isdir('./'+dir):
        if len(os.listdir('./'+dir))==1:# remove
            subprocess.run(['rm', '-rf', './'+dir])