import sys
import os
import commands
import subprocess

data_dir = '/tmp/tensorflow_pi'

def cmd(cmd):
    return commands.getoutput(cmd)
    # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # p.wait()
    # stdout, stderr = p.communicate()
    # return stdout.rstrip()

dirs = cmd("ls "+sys.argv[1])
backup_dir = os.path.dirname(os.path.abspath(__file__)) + "/model"
labels = dirs.splitlines()

# delete directories
if os.path.exists(data_dir):
    cmd("rm  -rf "+data_dir)

if os.path.exists(backup_dir):
    cmd("rm  -rf "+backup_dir)

# make directories
os.makedirs(data_dir+"/images")
os.makedirs(backup_dir)

#copy images and make train.txt/test.txt/label.txt
pwd = cmd('pwd')
imageDir = data_dir+"/images"
train = open(data_dir + '/train.txt','w')
train_lstm = open(data_dir + '/train_lstm.tsv','w')
test = open(data_dir + '/test.txt','w')
labelsTxt = open(data_dir + '/labels.txt','w')
labelsTxt_backup = open(backup_dir + '/labels.txt','w')

classNo=0
cnt = 0
#label = labels[classNo]
for label in labels:
    workdir = pwd+"/"+sys.argv[1]+"/"+label
    imageFiles = cmd("ls "+workdir+"/*.jpg")
    images = imageFiles.splitlines()
    print(label)
    labelsTxt.write(label+"\n")
    labelsTxt_backup.write(label+"\n")
    startCnt=cnt
    length = len(images)
    for image in images:
        imagepath = imageDir+"/image%07d" %cnt +".jpg"
        cmd("cp "+image+" "+imagepath)
        if cnt-startCnt < length*0.75:
            train.write(imagepath+" %d\n" % classNo)
            train_lstm.write(imagepath+"\t%s\n" % label)
        else:
            test.write(imagepath+" %d\n" % classNo)
        cnt += 1

    classNo += 1

print("class number=" + str(classNo))

train.close()
test.close()
train_lstm.close()
labelsTxt.close()
labelsTxt_backup.close()
