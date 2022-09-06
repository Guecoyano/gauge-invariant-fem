import  os, errno, sys

def PrintCopyright():
  print('---------------------------------------------------------------')
  print('Solving Boundary Value Problems (BVP\'s) with pyVecFEMP1 package')
  print('Copyright (C) 2015 Cuvelier F. and Scarella G.')
  print('  (LAGA/CNRS/University of Paris XIII)')
  print('---------------------------------------------------------------\n')

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False
  
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
      
def pause(message):
  if not run_from_ipython():
    if sys.version_info[0]==3 :
      input(message)
    else:
      raw_input(message)
