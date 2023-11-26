import os
import shutil
import math
import datetime

def create_output_dir(output_dir = "output"):
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    isotime = datetime.datetime.now().replace(microsecond=0).isoformat()
    output_path = os.path.join(output_dir, isotime.replace(':', '_'))
    os.mkdir(output_path)
    return output_path

def clear_temp_dir(temp_dir = 'temp'):
  if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

  for filename in os.listdir(temp_dir):
      file_path = os.path.join(temp_dir, filename)
      try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
          elif os.path.isdir(file_path):
              shutil.rmtree(file_path)
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))

def hline(title = ""):
  size = 80
  print()
  if len(title) > 0:
    l = math.floor((size - len(title)) / 2) - 1
    print(("-" * l) + " " + title + " " + ("-" * l))
  else:
    print("-" * size)
