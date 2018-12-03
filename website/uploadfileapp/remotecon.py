from paramiko import SSHClient
import paramiko
host="40.114.234.142"
user="aaron"
client = SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.load_system_host_keys()
client.connect(host, username=user, password="hadessleep10!")
stdin, stdout, stderr = client.exec_command('python3 hello_world.py')
print "stderr: ", stderr.readlines()
print "pwd: ", stdout.readlines()
