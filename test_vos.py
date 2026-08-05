
import vos as vos


fnames = ['/home/uname1/folder1/test1.tst',
          '/Users/uname1/folder1/test1.tst',
          'C:\\Users\\uname1\\folder1\\test1.tst',
          '~/Desktop/test1.tst',
          'folder1/test1.tst',
          'folder1\\test1.tst',
          'box/folder1/test1.tst',
          'box\\folder1\\test1.tst',
          'Box/folder1/test1.tst',
          'Box\\folder1\\test1.tst',
          '/Users/uname1/Library/CloudStorage/Box-Box/folder1/test1.tst',
          'C:\\Users\\uname1\\Box\\folder1\\test1.tst']


for fname in fnames:
    print(fname, "->",vos.file_path(fname))

#for fname in fnames:
#    print(fname, "->",vos.file_path(fname,box_root=True))

