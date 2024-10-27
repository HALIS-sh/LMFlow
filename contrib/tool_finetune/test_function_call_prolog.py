from pyswip import Prolog
import os

# Set the installation directory for SWI-Prolog
os.environ['SWI_HOME_DIR'] = '/home/wenhesun/anaconda3/envs/lmflow/lib/swipl'
# Set the dynamic link library path
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/home/wenhesun/anaconda3/envs/lmflow/lib'

# Create a Prolog instance
prolog = Prolog()
# Assert parent-child relationships
prolog.assertz("father(john, michael)")
prolog.assertz("father(michael, sarah)")

# Query the grandfather relationship
result = list(prolog.query("father(X, Y), father(Y, Z)"))
for solution in result:
    print(f"Grandfather: {solution['X']}, Grandchild: {solution['Z']}")
