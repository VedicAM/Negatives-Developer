import os
from developer import Developer
developer = Developer()

for filename in os.listdir('exampleDir'):
    f = os.path.join('exampleDir', filename)
    print(f)
    developer.write(f)