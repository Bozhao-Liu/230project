import os
	
files = [ file for file in os.listdir(os.getcwd()) if file.endswith(".py")]

for file in files:
	print(file)
	os.rename(file,file.replace("_", "-"))
