#! /bin/env python

'''
This is a draft script to find the modules imported into a script or notebook
and then list them, like in this Kaggle notebook: [Gensim Word2Vec Tutorial](https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial).
'''
import pkg_resources
import sys

modulenames = set(sys.modules) & set(globals())
allmodules = [sys.modules[name] for name in modulenames]
print(allmodules)

# Sample Output
'''
[<module 'gensim' from '/opt/local/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/gensim/__init__.py'>, <module 'pkg_resources' from '/opt/local/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/pkg_resources/__init__.py'>, <module 'types' from '/opt/local/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/types.py'>, <module 're' from '/opt/local/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/re.py'>, <module 'sys' (built-in)>]
'''

# This outputs 'pandas==1.0.3'
def version(module):
    version = pkg_resources.get_distribution(module).version
    return(f'{module}=={version}')

'''
A SO answer to this questions offered the following:

from modulefinder import ModuleFinder
finder = ModuleFinder()
finder.run_script("myscript.py")
for name, mod in finder.modules.items():
    print(name)

See: [How to list imported modules?](https://stackoverflow.com/questions/4858100/how-to-list-imported-modules)
'''