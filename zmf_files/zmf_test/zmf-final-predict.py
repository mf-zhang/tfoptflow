import subprocess

for i in range(1,5):
    print('python zmf_canon.py ' + str(i*5))
    subprocess.call('python zmf-canon.py ' + str(i*5), shell=True)