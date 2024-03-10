from genericpath import isdir, isfile
from operator import itemgetter
import os
import re
import sys

def addToFile(filename, replaceStr):
    print('Current working file', filename, '\n')
    try:   
        with open(filename, 'r') as f:
            file_content = f.read()
    except:
        try:
            with open(filename, 'r', encoding='gbk') as f:
                file_content = f.read()
        except IOError as e:
            print('error in opening file', filename,',', e)
            return    
    #os.remove(filename)
    with open(filename, 'w', encoding='utf8') as f:
        f.write(replaceStr + '\n\n' + file_content)

def deleteinfo(filename, replaceStr):
    print('Performing DELETE operation on', filename)    
    with open(filename, 'r', encoding='utf8') as f:
        file_content = f.read()
    file_content = file_content.replace(replaceStr, '')    
    file_content = deletespace(file_content)
    with open(filename, 'w', encoding='utf8') as f:                 
        f.write(file_content)

def deletespace(inputStr):
    pat = r'^\s+' 
    return re.sub(pat, '', inputStr)

# get the file paths with given extention, subfolders are included


def getFolderFiles(inpath: str, extname: str):
    filepaths = []
    for item in os.listdir(inpath):
        if os.path.isfile(os.path.join(inpath, item)) and os.path.splitext(item)[1] == extname:
            filepaths.append(os.path.join(inpath, item))
        elif os.path.isdir(os.path.join(inpath, item)):
            for subitem in getFolderFiles(os.path.join(inpath, item), extname):
                filepaths.append(subitem)
    return filepaths

#if True:  

print(getFolderFiles('D:/OneDrive/工作科研/Code/0_Github_upload/FSIPSO/src', '.java'))

if __name__== '__main__':   
    #print(os.listdir('.'))    
    cpath = os.path.dirname(os.path.abspath(__file__))
    print(cpath)
    #filenames = [x for x in os.listdir('.') if os.path.isfile(x) and os.path.splitext(x)[1]=='.m'] 
    filenames = getFolderFiles(os.getcwd(), '.java')
    print(filenames)  
    if len(sys.argv)>1 and sys.argv[1].lower() =='delete':
        action = deleteinfo        
    else:
        action = addToFile
    with open(cpath + '/addInfo.txt', 'r', encoding='utf8') as f:
        addstr = f.read()

    for filename in filenames:
        action(filename, addstr)



# if __name__== '__main__':   
#     #print(os.listdir('.'))
    
#     cpath = os.path.dirname(os.path.abspath(__file__))
#     print(cpath)
#     filenames = [x for x in os.listdir('.') if os.path.isfile(x) and os.path.splitext(x)[1]=='.m']    
#     if len(sys.argv)>1 and sys.argv[1].lower() =='delete':
#         action = deleteinfo        
#     else:
#         action = addToFile
#     with open(cpath + '/addInfo.txt', 'r') as f:
#         addstr = f.read()

#     for filename in filenames:
#         action(filename, addstr)
        #deleteinfo(filename, addstr)